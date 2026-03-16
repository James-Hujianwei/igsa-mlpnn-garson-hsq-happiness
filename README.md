# igsa-mlpnn-garson-hsq-happiness

Code and documentation for modeling the relationship between residents' happiness and human settlement quality using the IGSA-MLPNN-GARSON approach.

## Overview
This repository contains the source code, analysis scripts, and documentation associated with the manuscript:

**Modeling the Relationship Between Residents’ Happiness and Human Settlement Quality: An IGSA-MLPNN-GARSON Approach**

This package is a reproducible reference implementation of the workflow described in the manuscript. It covers data loading, preprocessing, MLPNN training, GSA/IGSA-based parameter optimization, 10-fold cross-validation, benchmark comparison, GARSON feature-importance analysis, and Fig. 10 generation.

The study is based on a questionnaire-derived human settlement quality (HSQ) evaluation system with 54 indicators and compares IGSA-MLPNN against multiple benchmark models, including MLPNN, GSA-MLPNN, SVM, LightGBM, RBFNN, Linear Regression, and Random Forest.

## Repository contents

```text
igsa-mlpnn-garson-hsq-happiness/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── run_all.py
├── run_fig10_only.py
└── src/
    ├── __init__.py
    ├── data.py
    ├── garson.py
    ├── metaheuristics.py
    ├── metrics.py
    ├── models.py
    ├── plotting.py
    └── utils.py
```

## Main outputs

Running the full pipeline generates the following main outputs:

- `cv10_fold_results.csv`
- `cv10_summary_numeric.csv`
- `Table1_accuracy_comparison_formatted.csv`
- `Table1_accuracy_comparison_formatted.xlsx`
- `Fig10_CV10_Comparison.png`
- `Fig10_CV10_Comparison.tif`
- `Fig10_CV10_Comparison.pdf`
- `garson_feature_importance.csv`
- `Garson_Top20.png`
- `Garson_Top20.tif`
- `Garson_Top20.pdf`
- `metaheuristic_best_params_by_fold.csv`
- `final_igsa_model_summary.json`

## Data access

The dataset underlying the findings of this study is publicly available in Zenodo:

**Dataset DOI:** [10.5281/zenodo.19037785](https://doi.org/10.5281/zenodo.19037785)

Please download the required dataset files from Zenodo and place them in your local working directory before running the code.

## Input data requirements

The scripts accept the following file formats:

- `.csv`
- `.xlsx`
- `.xls`
- `.parquet`

### Expected table format

- One row corresponds to one respondent
- One column corresponds to one feature / indicator
- One target column corresponds to residents’ happiness score
- All HSQ indicators should be numeric or convertible to numeric
- Missing values are mean-imputed within each fold during cross-validation

### Recommended column setup

- 54 HSQ indicators as feature columns
- 1 happiness target column, for example: `happiness`
- Optional metadata columns that can be excluded using `--drop-cols`

Example:

```text
X11, X12, ..., X84, happiness
```

## Environment

### Recommended Python version

- **Python 3.10**

### Main packages

- numpy
- pandas
- scipy
- scikit-learn
- tensorflow
- lightgbm
- matplotlib
- openpyxl

## Installation

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate igsa-mlpnn
```

### Option B: pip + virtual environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducibility

To reproduce the main analyses:

1. Download the dataset from Zenodo.
2. Place the required data files in your local working directory.
3. Install the required environment using `environment.yml` or `requirements.txt`.
4. Run the preprocessing, training, evaluation, and plotting scripts in the appropriate order.

## Run the full pipeline

```bash
python run_all.py \
  --data your_dataset.xlsx \
  --target happiness \
  --drop-cols respondent_id district subdistrict \
  --output-dir output
```

## Quick smoke test

```bash
python run_all.py \
  --data your_dataset.xlsx \
  --target happiness \
  --drop-cols respondent_id district subdistrict \
  --output-dir output_quick \
  --quick
```

## Reproduce the paper-style full setting

For a closer paper-style configuration consistent with the manuscript, you may use a 10-fold evaluation setting such as:

```bash
python run_all.py \
  --data your_dataset.xlsx \
  --target happiness \
  --drop-cols respondent_id district subdistrict \
  --output-dir output_full \
  --n-splits 10 \
  --gsa-pop 50 \
  --gsa-iters 200 \
  --epochs 200 \
  --batch-size 64 \
  --patience 20
```

## Regenerate Fig. 10 only

If you only need to regenerate Fig. 10 from summary values already reported in the manuscript, run:

```bash
python run_fig10_only.py
```

## Important notes for repository release and journal submission

1. Replace placeholder file paths with your actual dataset path.
2. Keep the final dataset schema consistent with the manuscript.
3. Use the same target column name in both code and manuscript.
4. If a journal requests the complete source code, provide the entire repository rather than only individual Python files.
5. Include both `requirements.txt` and `environment.yml`.
6. Keep this `README.md` in the repository and, if needed, also include it in supplementary materials.
7. If the public release does not include raw participant-level data, clearly state the data-access conditions while keeping the code fully available.

## Suggested release checklist

Before public release or journal submission, confirm that:

- [ ] `README.md` is included
- [ ] `requirements.txt` is included
- [ ] `environment.yml` is included
- [ ] `run_all.py` is included
- [ ] `run_fig10_only.py` is included
- [ ] `src/` modules are included
- [ ] The manuscript target column name matches the code
- [ ] Baseline model names match the manuscript tables
- [ ] Fig. 10 can be regenerated successfully
- [ ] Table 1 can be regenerated successfully
- [ ] GARSON importance results can be exported successfully
- [ ] No hard-coded local absolute paths remain
- [ ] No private credentials, tokens, or personal paths remain

## Reproducibility note

Because model initialization and metaheuristic optimization contain stochastic components, this package fixes random seeds where possible. Small numerical deviations may still occur across different hardware environments, TensorFlow versions, and BLAS backends.

## Citation

If you use this repository, please cite the associated article and the archived software release.

**Software DOI:** [10.5281/zenodo.19048156](https://doi.org/10.5281/zenodo.19048156)

## License

This repository is released under the MIT License.
