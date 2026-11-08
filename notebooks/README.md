# BAF analysis notebooks

- `eda_A2-09.ipynb`: class balance, missing values, category/numeric comparisons and temporal drift.
- `isolation_forest_A2-13.ipynb`: Isolation Forest compared with a historical logistic-regression baseline from commit `661871e` of `KennyDao1324/ai4cyber-fraud`.
- `a2_analysis/`: reusable helpers. `a2_outputs/`: saved figures and result tables.

Use the pinned environment in `../experiment_suite/` and install Jupyter if required by your editor. Put the original BAF Base CSV in `../experiment_suite/dataset/baf/Base.csv`. The prepared final-model CSV is not a substitute for the complete raw data used by these notebooks. Run cells in order; rerunning regenerates notebook analysis outputs, not production models.

The historical comparison is not the current v4 CatBoost champion. It intentionally retains the older baseline's missing-value handling and fold design. Current production handling treats all negative `intended_balcon_amount` as missing, and its age fairness audit uses `>=50`. Existing notebook outputs are historical evidence; narrative references to earlier pipeline limitations must be read in that context.

Recall at 5% FPR in this study is an ROC-derived diagnostic using validation labels, not an independently calibrated deployment operating point. Isolation Forest's training-contamination cutoff is a separate measure. Neither should be substituted for the selected model's calibrated results. These notebooks use months 0–5 for model comparisons; other experiments have already inspected months 6–7, so this does not establish a new blind test. K-means clustering and the four-family supervised comparison remain separate and unchanged.

Source: [BAF dataset documentation](https://github.com/feedzai/bank-account-fraud). Data licence: CC BY-NC-SA 4.0. Original authorship and source references are retained.
