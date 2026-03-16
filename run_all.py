from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.data import load_table, prepare_xy
from src.garson import garson_importance
from src.metaheuristics import ObjectiveWrapper, SearchSpace, gsa_optimize, igsa_optimize
from src.metrics import mean_ci, regression_metrics
from src.models import KerasMLPRegressor, MLPConfig, build_baseline_model
from src.plotting import plot_fig10, plot_top_garson
from src.utils import ensure_dir, save_json, set_seed


META_MODELS = {"MLPNN", "GSA-MLPNN", "IGSA-MLPNN"}
BASELINE_MODELS = ["LR", "RF", "SVM", "LightGBM", "RBFNN"]
ALL_MODELS = ["IGSA-MLPNN", "GSA-MLPNN", "MLPNN", "SVM", "LightGBM", "RBFNN", "LR", "RF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IGSA-MLPNN-GARSON paper code submission package")
    parser.add_argument("--data", required=True, help="Path to CSV/XLSX/XLS/Parquet data file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--drop-cols", nargs="*", default=[], help="Columns to exclude from feature matrix")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--gsa-pop", type=int, default=20)
    parser.add_argument("--gsa-iters", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Use smaller settings for a smoke test")
    return parser.parse_args()


def preprocess_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SimpleImputer, MinMaxScaler, MinMaxScaler]:
    imp = SimpleImputer(strategy="mean")
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)

    X_train_scaled = x_scaler.fit_transform(X_train_imp)
    X_test_scaled = x_scaler.transform(X_test_imp)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, imp, x_scaler, y_scaler


def build_mlp_config(params: Dict[str, float], args: argparse.Namespace, seed: int) -> MLPConfig:
    return MLPConfig(
        hidden_units=int(params["hidden_units"]),
        learning_rate=float(params["learning_rate"]),
        init_range=float(params["init_range"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=0,
        seed=seed,
    )


def evaluate_mlp_params(
    params: Dict[str, float],
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> float:
    X_sub, X_val, y_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    model = KerasMLPRegressor(build_mlp_config(params, args, seed))
    model.fit(X_sub, y_sub, validation_data=(X_val, y_val))
    preds = np.clip(model.predict(X_val), 0.0, 1.0)
    return regression_metrics(y_val, preds)["MAE"]


def train_predict_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, Dict[str, float], object]:
    if model_name == "MLPNN":
        params = {"hidden_units": 10, "learning_rate": 0.01, "init_range": 0.5}
        model = KerasMLPRegressor(build_mlp_config(params, args, seed))
        X_sub, X_val, y_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
        model.fit(X_sub, y_sub, validation_data=(X_val, y_val))
        preds = np.clip(model.predict(X_test), 0.0, 1.0)
        return preds, params, model

    if model_name in {"GSA-MLPNN", "IGSA-MLPNN"}:
        space = SearchSpace()
        objective = ObjectiveWrapper(
            lambda p: evaluate_mlp_params(p, X_train, y_train, seed=seed, args=args),
            search_space=space,
        )
        if model_name == "GSA-MLPNN":
            best_params, _, _ = gsa_optimize(
                objective,
                pop_size=args.gsa_pop,
                iterations=args.gsa_iters,
                seed=seed,
            )
        else:
            best_params, _, _ = igsa_optimize(
                objective,
                pop_size=args.gsa_pop,
                iterations=args.gsa_iters,
                seed=seed,
            )
        model = KerasMLPRegressor(build_mlp_config(best_params, args, seed))
        X_sub, X_val, y_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
        model.fit(X_sub, y_sub, validation_data=(X_val, y_val))
        preds = np.clip(model.predict(X_test), 0.0, 1.0)
        return preds, best_params, model

    estimator = build_baseline_model(model_name, random_state=seed)
    estimator.fit(X_train, y_train)
    preds = np.clip(np.asarray(estimator.predict(X_test)).ravel(), 0.0, 1.0)
    return preds, {}, estimator


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    igsa = fold_results[fold_results["Model"] == "IGSA-MLPNN"].copy()

    for model in ALL_MODELS:
        dfm = fold_results[fold_results["Model"] == model].copy()
        row: Dict[str, float | str] = {"Model": model}
        for metric in ["MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE"]:
            vals = dfm[metric].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_mean, ci_low, ci_high = mean_ci(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
            row[f"{metric}_ci_half"] = ci_high - ci_mean

        if model == "IGSA-MLPNN":
            row["p_value_MAE"] = np.nan
            row["p_value_R2"] = np.nan
        else:
            merged = igsa[["Fold", "MAE", "R2"]].merge(
                dfm[["Fold", "MAE", "R2"]],
                on="Fold",
                suffixes=("_IGSA", "_OTHER"),
            )
            row["p_value_MAE"] = float(stats.ttest_rel(merged["MAE_IGSA"], merged["MAE_OTHER"]).pvalue)
            row["p_value_R2"] = float(stats.ttest_rel(merged["R2_IGSA"], merged["R2_OTHER"]).pvalue)
        rows.append(row)

    return pd.DataFrame(rows)


def format_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Model": summary_df["Model"]})
    out["MAE"] = summary_df.apply(lambda r: f"{r['MAE_mean']:.4f} ± {r['MAE_sd']:.4f} [{r['MAE_ci_low']:.4f}, {r['MAE_ci_high']:.4f}]", axis=1)
    out["MSE"] = summary_df.apply(lambda r: f"{r['MSE_mean']:.5f} ± {r['MSE_sd']:.5f}", axis=1)
    out["RMSE"] = summary_df.apply(lambda r: f"{r['RMSE_mean']:.4f} ± {r['RMSE_sd']:.4f}", axis=1)
    out["R²"] = summary_df.apply(lambda r: f"{r['R2_mean']:.4f} ± {r['R2_sd']:.4f} [{r['R2_ci_low']:.4f}, {r['R2_ci_high']:.4f}]", axis=1)
    out["RMSLE"] = summary_df.apply(lambda r: f"{r['RMSLE_mean']:.4f} ± {r['RMSLE_sd']:.4f}", axis=1)
    out["MAPE"] = summary_df.apply(lambda r: f"{r['MAPE_mean']:.1f}% ± {r['MAPE_sd']:.1f}%", axis=1)
    out["p-value (MAE)"] = summary_df["p_value_MAE"].apply(lambda x: "—" if pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}"))
    out["p-value (R²)"] = summary_df["p_value_R2"].apply(lambda x: "—" if pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}"))
    return out


def fit_final_igsa_model(X: pd.DataFrame, y: pd.Series, args: argparse.Namespace, feature_names: List[str]):
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3, random_state=args.seed)
    X_train_s, X_test_s, y_train_s, y_test_s, imp, x_scaler, y_scaler = preprocess_fold(X_train, X_test, y_train, y_test)

    space = SearchSpace()
    objective = ObjectiveWrapper(
        lambda p: evaluate_mlp_params(p, X_train_s, y_train_s, seed=args.seed, args=args),
        search_space=space,
    )
    best_params, best_fitness, history = igsa_optimize(
        objective, pop_size=args.gsa_pop, iterations=args.gsa_iters, seed=args.seed
    )
    model = KerasMLPRegressor(build_mlp_config(best_params, args, args.seed))
    X_sub, X_val, y_sub, y_val = train_test_split(X_train_s, y_train_s, test_size=0.2, random_state=args.seed)
    model.fit(X_sub, y_sub, validation_data=(X_val, y_val))
    preds = np.clip(model.predict(X_test_s), 0.0, 1.0)
    metrics = regression_metrics(y_test_s, preds)
    w1, w2 = model.get_weight_matrices()
    garson_df = garson_importance(feature_names, w1, w2)

    extras = {
        "best_params": best_params,
        "best_validation_mae": float(best_fitness),
        "holdout_metrics_scaled": metrics,
        "search_history": history,
    }
    return garson_df, extras


def main() -> None:
    args = parse_args()
    if args.quick:
        args.gsa_pop = min(args.gsa_pop, 8)
        args.gsa_iters = min(args.gsa_iters, 8)
        args.epochs = min(args.epochs, 50)
        args.patience = min(args.patience, 8)

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    df = load_table(args.data)
    X, y = prepare_xy(df, args.target, drop_cols=args.drop_cols)
    feature_names = X.columns.tolist()

    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_rows: List[Dict[str, float | int | str]] = []
    best_params_log: List[Dict[str, float | int | str]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_np), start=1):
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]
        X_train_s, X_test_s, y_train_s, y_test_s, *_ = preprocess_fold(X_train, X_test, y_train, y_test)

        fold_seed = args.seed + fold_idx
        for model_name in ALL_MODELS:
            preds, params, _ = train_predict_model(model_name, X_train_s, y_train_s, X_test_s, args, fold_seed)
            metrics = regression_metrics(y_test_s, preds)
            row: Dict[str, float | int | str] = {"Fold": fold_idx, "Model": model_name, **metrics}
            fold_rows.append(row)
            if params:
                best_params_log.append({"Fold": fold_idx, "Model": model_name, **params})
            print(f"Fold {fold_idx:02d} | {model_name:12s} | MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.4f}")

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(output_dir / "cv10_fold_results.csv", index=False)

    if best_params_log:
        pd.DataFrame(best_params_log).to_csv(output_dir / "metaheuristic_best_params_by_fold.csv", index=False)

    summary_df = summarize_results(fold_results)
    summary_df.to_csv(output_dir / "cv10_summary_numeric.csv", index=False)

    formatted = format_summary_table(summary_df)
    formatted.to_csv(output_dir / "Table1_accuracy_comparison_formatted.csv", index=False)
    formatted.to_excel(output_dir / "Table1_accuracy_comparison_formatted.xlsx", index=False)

    plot_fig10(summary_df, output_dir)

    garson_df, extras = fit_final_igsa_model(X, y, args, feature_names)
    garson_df.to_csv(output_dir / "garson_feature_importance.csv", index=False)
    plot_top_garson(garson_df, output_dir, top_n=20)
    save_json(extras, output_dir / "final_igsa_model_summary.json")

    print("\nAll outputs have been saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
