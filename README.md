# Bank Account Fraud Detection

A Python project for fraud-data analysis, supervised classification and exploratory clustering. It includes trained pipelines, a local dashboard, evaluation charts and reproducible training code.

## Features

- Four supervised classifiers: logistic regression, random forest, extra trees, catboost.
- Separate MiniBatch K-means segmentation with training-only selection and seed-stability checks.
- Time-ordered data splits, validation-calibrated alert thresholds and saved model pipelines.
- FastAPI dashboard for predictions, metrics and downloadable evaluation files.

## Quick start

```bash
git clone --branch baf --single-branch https://github.com/KennyDao1324/fraud-detection-datasets.git
cd fraud-detection-datasets/experiment_suite
conda env create -f environment.yml
conda activate fraudsuite
python app.py
```

Open http://127.0.0.1:8050/. Saved models work without downloading the raw data. Use the example engineered record shown in the dashboard; raw transaction rows may require preprocessing.

In PyCharm, open the project folder and select the `fraudsuite` conda environment as interpreter. Training uses Mac/Linux file locks; inference does not require them.

## Project structure

| Path | Contents |
|---|---|
| `experiment_suite/training_model/` | Current preprocessing, classification and clustering code |
| `experiment_suite/pipelines/baf/` | Dataset-specific implementation and configuration |
| `experiment_suite/evaluation/baf/` | Saved pipelines, metrics, charts and model cards |
| `experiment_suite/dataset/baf/` | Local raw input files; excluded from Git |
| `experiment_suite/app.py` | Local API and dashboard server |
| `training_model/train.py` | Classification launcher |
| `training_model/cluster.py` | Clustering launcher |
| `src/`, `scripts/`, `docs/` | Original dataset implementation, utilities and provenance |
| `artifacts/`, `checkpoint/`, `outputs/` | Earlier baseline models and evaluation evidence |
| `notebooks/eda_A2-09.ipynb`, `notebooks/isolation_forest_A2-13.ipynb` | BAF exploration and historical Isolation Forest vs logistic-regression diagnostics, with stored outputs; see `notebooks/README.md`; require original `Base.csv` in `experiment_suite/dataset/baf/` |

Small `__init__.py`, `.gitkeep` and launcher files support packaging or folder layout; the main implementation lives in the source and training modules.

## Classification results

Experiment: `feature_policy_v4`. Selected model: **catboost**. The table uses the 5% calibration-FPR headline operating point; actual later-period FPR can differ. 1% is secondary.

| Model | PR-AUC | Recall | FPR | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.1746 | 44.7% | 3.160% | 16.8% | 0.2440 | 0.8952 |
| random_forest | 0.1534 | 38.3% | 2.976% | 15.5% | 0.2207 | 0.8816 |
| extra_trees | 0.1525 | 36.3% | 2.889% | 15.2% | 0.2142 | 0.8744 |
| catboost | 0.2127 | 51.1% | 3.584% | 16.9% | 0.2539 | 0.9066 |

[Model comparison](experiment_suite/evaluation/baf/feature_policy_v4/model_comparison.png) · [PR curves](experiment_suite/evaluation/baf/feature_policy_v4/precision_recall_curves.png) · [Metrics CSV](experiment_suite/evaluation/baf/feature_policy_v4/model_comparison.csv) · [Full results and protocol](experiment_suite/evaluation/baf/feature_policy_v4/report.json)

Model settings and thresholds were selected using chronological validation. Test periods have already been inspected, so comparisons are exploratory rather than a fresh holdout. A score supports review and does not establish fraud or authorize an account decision.

[Model card](experiment_suite/evaluation/baf/feature_policy_v4/MODEL_CARD.md) · [Threshold-policy comparison](experiment_suite/evaluation/baf/feature_policy_v4/threshold_policies.csv)

The dashboard supports five calibration FPR budgets, optional review-capacity limits and an approximate one-sided 95% calibration FPR bound. These are calibration diagnostics, not guarantees of future FPR or alert volume.

## Clustering

MiniBatch K-means selected **k=5**. Training silhouette: **0.067**; mean seed ARI: **0.347**.

Weak or unstable geometry; no candidate met both size and repeatability floors.

Preprocessing and k selection use training inputs only. Fraud labels and label-derived features are excluded from the inputs and used only afterward to interpret cluster fraud rates. Clusters are not fraud predictions; silhouette is not classification accuracy.

[Interpretation](experiment_suite/evaluation/baf/clustering_extended_v2/README.md) · [Silhouette and stability](experiment_suite/evaluation/baf/clustering_extended_v2/silhouette_stability.png) · [Cluster sizes](experiment_suite/evaluation/baf/clustering_extended_v2/cluster_sizes.png) · [Feature profiles](experiment_suite/evaluation/baf/clustering_extended_v2/feature_profiles.png) · [Fraud-rate interpretation](experiment_suite/evaluation/baf/clustering_extended_v2/fraud_rates.png)

## Training

Place the original downloaded files in `experiment_suite/dataset/baf/`. Keep the raw files local and follow their source licences. From `experiment_suite/`, prepare the data:

```bash
python -c "import sys; sys.path.insert(0, 'training_model'); from priority_data import prepare; prepare('baf')"
python training_model/feature_policy_search.py --dataset baf
python training_model/clustering.py --dataset baf
```

Completed experiment folders are preserved and may report already complete or reuse cached fits. Use a separate working copy and a new versioned output directory for a new search; do not overwrite the published evidence. Run classification and clustering separately.

## Dataset and licence

- [Bank Account Fraud Dataset Suite (NeurIPS 2022) on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- [Official BAF documentation, datasheet, and citation](https://github.com/feedzai/bank-account-fraud)

The BAF data is provided under CC BY-NC-SA 4.0. Keep the source, licence, and citation with any analysis
or derived dataset, and do not commit the large raw CSV files to this repository.


Raw datasets and row-level prediction exports are not redistributed here. Code and dataset licences are separate; check the original source terms before reuse.

[Selected-model identity and operating points](FINAL_MODEL.md) · [Conda workflow](experiment_suite/CONDA_GUIDE.md)

## Prepared data and reproducibility

[CSV export, scoring, refitting and prediction commands](experiment_suite/CONDA_GUIDE.md). Each export has a data dictionary and checksum manifest. Refit outputs are separate from the selected checkpoint.
