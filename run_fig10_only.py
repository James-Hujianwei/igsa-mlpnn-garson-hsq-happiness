from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.plotting import plot_fig10


# Replace with your final numerical summary if needed.
TABLE = [
    ["IGSA-MLPNN", 0.0531, 0.00309, 0.0556, 0.9652, 0.0117, 5.6, 0.0514, 0.0548, 0.9608, 0.9696],
    ["GSA-MLPNN", 0.0592, 0.00383, 0.0619, 0.9487, 0.0130, 6.2, 0.0571, 0.0613, 0.9434, 0.9540],
    ["MLPNN", 0.0683, 0.00509, 0.0713, 0.9251, 0.0150, 7.1, 0.0651, 0.0715, 0.9178, 0.9324],
    ["SVM", 0.0766, 0.00635, 0.0797, 0.8812, 0.0167, 7.9, 0.0730, 0.0802, 0.8713, 0.8911],
    ["LightGBM", 0.0783, 0.00669, 0.0818, 0.8653, 0.0172, 8.1, 0.0740, 0.0826, 0.8549, 0.8757],
    ["RBFNN", 0.0755, 0.00619, 0.0787, 0.8723, 0.0165, 7.8, 0.0721, 0.0789, 0.8633, 0.8813],
    ["LR", 0.0849, 0.00738, 0.0859, 0.8468, 0.0184, 8.8, 0.0803, 0.0896, 0.8358, 0.8578],
    ["RF", 0.0724, 0.00574, 0.0758, 0.9086, 0.0158, 7.4, 0.0690, 0.0758, 0.9004, 0.9168],
]


def main() -> None:
    df = pd.DataFrame(TABLE, columns=[
        "Model", "MAE_mean", "MSE_mean", "RMSE_mean", "R2_mean", "RMSLE_mean", "MAPE_mean",
        "MAE_ci_low", "MAE_ci_high", "R2_ci_low", "R2_ci_high"
    ])
    for col in ["MSE", "RMSE", "RMSLE", "MAPE"]:
        df[f"{col}_ci_half"] = 0.0
    df["MAE_ci_half"] = df["MAE_ci_high"] - df["MAE_mean"]
    df["R2_ci_half"] = df["R2_ci_high"] - df["R2_mean"]
    out_dir = Path("output_fig10_only")
    out_dir.mkdir(exist_ok=True, parents=True)
    plot_fig10(df, out_dir)
    print(f"Figure saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
