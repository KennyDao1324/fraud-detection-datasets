"""
isoforest.py - Isolation Forest helpers for the BAF Base dataset.

Isolation Forest anomaly detection compared with a historical supervised
baseline. Author: Thong. Reviewer: Phuc.

The supervised comparison model is the team's own classifier from A2-11
(Kha's repo KennyDao1324/ai4cyber-fraud, commit 661871e): the same
preprocessing, features and logistic regression settings, re-fitted here on
each time-ordered fold so it never sees months 6-7. Fold C below (train
months 0-4, validate month 5) is the team's own train/validation split.

The comparison preserves that historical baseline for reproducibility. It
is not the currently selected v4 CatBoost model. The current production
pipeline masks all negative intended_balcon_amount values and uses age >= 50
for its fairness audit; this notebook's older baseline is intentionally separate.

Structure of this module
  1. Constants          - seed, time-ordered validation folds, grids
  2. Features           - preprocessing pipeline (fitted on training months only)
  3. Metrics            - PR-AUC, recall / precision at 5% FPR, flag-rate metrics
  4. Models             - Isolation Forest, the team's logistic regression
  5. Experiments        - hyper-parameter grid, contamination sweep, overlap
  6. Figures            - one plot_* function per report figure

The notebook notebooks/isolation_forest_A2-13.ipynb calls these functions.
This study uses months 0-5 only. Other repository experiments have already
inspected months 6-7; they must not be described as a fresh holdout.
Recall at 5% FPR here is a retrospective ROC diagnostic using validation
labels, not a deployment threshold calibrated on a separate period.
"""

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import eda

# ===========================================================================
# 1. Constants
# ===========================================================================

SEED = 30049                     # same seed as the rest of the project

# Time-ordered ("expanding window") validation inside the development months:
# train on every month before the validation month, never on later ones.
# Months 3 and 4 are used to choose settings; month 5 (train 0-4) is held back
# and looked at once, so the reported result is not tuned on itself.
TUNE_VAL_MONTHS = (3, 4)
FINAL_VAL_MONTH = 5

# Isolation Forest grid. `contamination` is NOT in this grid: it only moves the
# decision threshold, never the ranking (see tune_contamination).
IF_GRID = {
    "feature_set": ["numeric", "with_categorical"],
    "n_estimators": [100, 300],
    "max_samples": [256, 1024],
    "max_features": [0.5, 1.0],
}
CONTAMINATION_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
SEEDS_FOR_SPREAD = [30049, 1, 2, 3, 4]

# Rarely seen category levels are pooled by the one-hot encoder.
MIN_LEVEL_COUNT = 1000

# Team model settings, copied from the team repo (configs/training.json, data.py,
# training.py) so the comparison is like for like.
TEAM_MODEL = "Logistic regression (team repo)"
TEAM_SEED = 42
TEAM_TRAIN_ROWS = 250_000        # stratified (month x label) training sample
# Never used as model inputs (OOS07 / team rule); device_fraud_count is constant.
PROTECTED_FEATURES = ["customer_age", "income", "employment_status"]
# Columns where the team repo turns the BAF code -1 into NaN before modelling
# (MISSING_SENTINEL_COLUMNS in the repo's constants.py).
TEAM_SENTINEL_COLUMNS = ["prev_address_months_count", "current_address_months_count",
                         "intended_balcon_amount", "bank_months_count",
                         "session_length_in_minutes", "device_distinct_emails_8w"]

MODEL_STYLE = {                  # colour + line style stay distinct in greyscale
    "Isolation Forest": dict(color="#B2182B", ls="-", lw=1.6),
    TEAM_MODEL: dict(color="#08306B", ls="--", lw=1.4),
}


# ===========================================================================
# 2. Features
# ===========================================================================

def feature_columns(df):
    """Return (numeric, flags, categorical) column lists used as model inputs.

    Excludes the target, `month` (a time index, not a customer attribute),
    constant columns such as device_fraud_count, and the protected features
    (age, income, employment status), exactly as the team classifier does.
    """
    numeric = [c for c in eda.numeric_features(df) if c not in PROTECTED_FEATURES]
    flags = [c for c in eda.BINARY_FLAGS if c in df.columns]
    categorical = [c for c in eda.CATEGORICAL if c not in PROTECTED_FEATURES]
    return numeric, flags, categorical


def make_preprocessor(df, feature_set="numeric", scale=False):
    """Build the preprocessing step (unfitted; fit it on training months only).

    - BAF missing codes (-1 / negative) have already been turned into NaN by
      eda.with_missing_as_nan; here NaN is filled with the training median and
      a missing-indicator column is added for every column that has gaps
      (missingness itself carries signal, see EDA Finding 5).
    - feature_set "numeric": numeric features and 0/1 flags only.
    - feature_set "with_categorical": also one-hot encodes the 5 categoricals.
    - scale=True standardises numeric columns (needed for logistic regression).
    """
    numeric, flags, categorical = feature_columns(df)
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    blocks = [("num", Pipeline(steps), numeric + flags)]
    if feature_set == "with_categorical":
        blocks.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                            min_frequency=MIN_LEVEL_COUNT,
                                            sparse_output=False), categorical))
    elif feature_set != "numeric":
        raise ValueError(f"unknown feature_set: {feature_set!r}")
    return ColumnTransformer(blocks)


def load_dev(path, missing_as_nan=True):
    """Load Base.csv and keep development months 0-5 only.

    Months 6-7 are dropped here so nothing downstream can touch them.
    missing_as_nan=True turns the BAF missing codes (-1 / negative) into NaN
    (used by the forest); False keeps the raw codes (the team classifier
    reproduces the historical baseline's -1-only missing-code conversion).
    """
    df = eda.load_base(path)
    dev = df[df["month"].isin(eda.DEV_MONTHS)].copy()
    return eda.with_missing_as_nan(dev) if missing_as_nan else dev


def split_by_month(df, train_months, val_month):
    """Return (train, validation) frames; train months must precede validation."""
    if max(train_months) >= val_month:
        raise ValueError("training months must all come before the validation month")
    return df[df["month"].isin(train_months)], df[df["month"] == val_month]


def prepare_fold(df, val_month, feature_set, scale=False):
    """Fit the preprocessor on months < val_month and transform both parts.

    Returns X_train, y_train, X_val, y_val as arrays (float32 features).
    """
    train, val = split_by_month(df, range(val_month), val_month)
    prep = make_preprocessor(df, feature_set, scale=scale)
    X_train = prep.fit_transform(train).astype("float32")
    X_val = prep.transform(val).astype("float32")
    return X_train, train[eda.TARGET].to_numpy(), X_val, val[eda.TARGET].to_numpy()


# ===========================================================================
# 3. Metrics
# ===========================================================================

def recall_at_fpr(y, score, fpr=eda.TARGET_FPR):
    """Recall (and precision, threshold) at the best point with FPR <= `fpr`.

    The BAF benchmark metric. The threshold is the score cut-off that lets at
    most `fpr` of the legitimate applications through the alarm.
    """
    f, t, thr = roc_curve(y, score)
    i = np.where(f <= fpr)[0].max()
    flagged = score >= thr[i]
    precision = y[flagged].mean() if flagged.any() else np.nan
    return {"recall_at_5fpr": t[i], "precision_at_5fpr": precision,
            "threshold_at_5fpr": thr[i], "achieved_fpr": f[i]}


def evaluate(y, score):
    """Threshold-free metrics plus recall/precision at 5% FPR for one score."""
    out = {"pr_auc": average_precision_score(y, score),
           "roc_auc": roc_auc_score(y, score)}
    out.update(recall_at_fpr(y, score))
    return out


def flag_metrics(y, score, threshold):
    """Precision, recall, F1 and flag rate when `score >= threshold` is flagged."""
    flagged = score >= threshold
    tp = int((flagged & (y == 1)).sum())
    precision = tp / flagged.sum() if flagged.any() else 0.0
    recall = tp / (y == 1).sum()
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "flag_rate": flagged.mean()}


# ===========================================================================
# 4. Models
# ===========================================================================

def anomaly_score(model, X):
    """Isolation Forest anomaly score; higher = more anomalous (sklearn's
    score_samples is the negative of this)."""
    return -model.score_samples(X)


def fit_isolation_forest(X_train, n_estimators=100, max_samples=256,
                         max_features=1.0, seed=SEED, contamination="auto"):
    """Fit an unsupervised Isolation Forest. Labels are never used."""
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            max_features=max_features, contamination=contamination,
                            random_state=seed, n_jobs=2)
    return model.fit(X_train)


def team_columns(raw):
    """Input columns of the team classifier, in the team repo's column order.

    Everything except the target, `month`, the protected features and the
    constant device_fraud_count. Categorical = text columns (team rule).
    """
    excluded = {eda.TARGET, "month", *PROTECTED_FEATURES, "device_fraud_count"}
    features = [c for c in raw.columns if c not in excluded]
    categorical = [c for c in features if c in eda.CATEGORICAL]
    numeric = [c for c in features if c not in categorical]
    return features, numeric, categorical


def team_pipeline(numeric, categorical):
    """The team's A2-11 logistic regression pipeline (repo commit 661871e).

    Numeric: median imputation + missing indicators + standardisation (the
    -1 codes are turned into NaN beforehand, see team_frame). Categorical:
    most-frequent imputation + one-hot encoding. Model: class-balanced
    logistic regression (lbfgs, 2000 iterations).
    """
    numeric_steps = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scaler", StandardScaler())])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                  sparse_output=True))])
    prep = ColumnTransformer([("num", numeric_steps, numeric),
                              ("cat", categorical_steps, categorical)],
                             remainder="drop", sparse_threshold=1.0)
    model = LogisticRegression(class_weight="balanced", max_iter=2000,
                               solver="lbfgs", random_state=TEAM_SEED)
    return Pipeline([("preprocessor", prep), ("model", model)])


def team_sample(train, limit=TEAM_TRAIN_ROWS, seed=TEAM_SEED):
    """The team's training sample: `limit` rows stratified by month x label."""
    if len(train) <= limit:
        return train.reset_index(drop=True)
    strata = train["month"].astype(str) + "_" + train[eda.TARGET].astype(str)
    sampled, _ = train_test_split(train, train_size=limit, random_state=seed,
                                  stratify=strata)
    return sampled.sort_index().reset_index(drop=True)


def team_frame(raw):
    """Copy of `raw` prepared the way the team repo prepares its data: float64
    numbers, plain text categoricals, and -1 turned into NaN in the six
    sentinel columns (only -1, unlike eda.with_missing_as_nan)."""
    out = raw.copy()
    for col in out.columns:
        if col in eda.CATEGORICAL:
            out[col] = out[col].astype(object)
        elif out[col].dtype.kind == "f":
            out[col] = out[col].astype("float64")
    for col in TEAM_SENTINEL_COLUMNS:
        out[col] = out[col].mask(out[col].eq(-1))
    return out


def fit_team_model(raw, val_month):
    """Re-fit the team logistic regression on months < val_month; score val_month.

    `raw` must keep the BAF missing codes (load_dev(..., missing_as_nan=False));
    the team's -1 to NaN step is applied inside.
    Returns (y_val, scores). The team's frozen model is trained on months 0-5;
    here it is re-fitted on earlier months only so it can be compared on the
    same time-ordered folds as the forest without touching months 6-7.
    """
    frame = team_frame(raw)
    _, numeric, categorical = team_columns(frame)
    train = team_sample(frame[frame["month"] < val_month])
    val = frame[frame["month"] == val_month]
    pipe = team_pipeline(numeric, categorical).fit(train[numeric + categorical],
                                                   train[eda.TARGET])
    return (val[eda.TARGET].to_numpy(),
            pipe.predict_proba(val[numeric + categorical])[:, 1])


# ===========================================================================
# 5. Experiments
# ===========================================================================

def tune_isolation_forest(dev, val_months=TUNE_VAL_MONTHS, grid=None):
    """Grid search over Isolation Forest settings on time-ordered folds.

    For every (feature_set, n_estimators, max_samples, max_features) and every
    validation month: fit on all earlier months (no labels), score the
    validation month, record PR-AUC and recall at 5% FPR. Returns one row per
    combination and fold.
    """
    grid = grid or IF_GRID
    rows = []
    for val_month in val_months:
        for fset in grid["feature_set"]:
            X_tr, _, X_va, y_va = prepare_fold(dev, val_month, fset)
            combos = itertools.product(grid["n_estimators"], grid["max_samples"],
                                       grid["max_features"])
            for n_est, max_s, max_f in combos:
                t0 = time.time()
                model = fit_isolation_forest(X_tr, n_est, max_s, max_f)
                m = evaluate(y_va, anomaly_score(model, X_va))
                rows.append({"val_month": val_month, "feature_set": fset,
                             "n_estimators": n_est, "max_samples": max_s,
                             "max_features": max_f, "pr_auc": m["pr_auc"],
                             "roc_auc": m["roc_auc"],
                             "recall_at_5fpr": m["recall_at_5fpr"],
                             "seconds": time.time() - t0})
    return pd.DataFrame(rows)


def best_settings(results):
    """Average the grid over folds and return (summary table, best row).

    Ranking quality (mean PR-AUC) decides; recall at 5% FPR breaks ties.
    """
    keys = ["feature_set", "n_estimators", "max_samples", "max_features"]
    summary = (results.groupby(keys, as_index=False)
               [["pr_auc", "roc_auc", "recall_at_5fpr"]].mean()
               .sort_values(["pr_auc", "recall_at_5fpr"], ascending=False)
               .reset_index(drop=True))
    return summary, summary.iloc[0]


def threshold_for_rate(train_scores, contamination):
    """Score cut-off that flags the top `contamination` share of training rows.

    This is exactly what sklearn does with `contamination=c`: it sets the
    decision offset to the (1 - c) quantile of the training anomaly scores.
    """
    return np.quantile(train_scores, 1 - contamination)


def tune_contamination(dev, settings, val_months=TUNE_VAL_MONTHS,
                       grid=CONTAMINATION_GRID):
    """Sweep `contamination` for the chosen settings on the tuning folds.

    The score ranking (so PR-AUC and recall at 5% FPR) does not depend on
    contamination; it only sets where the yes/no cut-off falls, so precision,
    recall and F1 at that cut-off change. Threshold comes from the training
    months only; the metrics come from the validation month.
    """
    rows = []
    for val_month in val_months:
        X_tr, _, X_va, y_va = prepare_fold(dev, val_month, settings["feature_set"])
        model = fit_isolation_forest(X_tr, int(settings["n_estimators"]),
                                     int(settings["max_samples"]),
                                     float(settings["max_features"]))
        train_scores, val_scores = anomaly_score(model, X_tr), anomaly_score(model, X_va)
        pr_auc = average_precision_score(y_va, val_scores)
        for c in grid:
            m = flag_metrics(y_va, val_scores, threshold_for_rate(train_scores, c))
            rows.append({"val_month": val_month, "contamination": c,
                         "pr_auc": pr_auc, **m})
    return pd.DataFrame(rows)


def choose_contamination(f1_by_c, tolerance=0.05):
    """Smallest contamination whose mean F1 is within `tolerance` of the best.

    With a weak ranker F1 keeps creeping up as more applications are flagged,
    so the raw maximum lands on the edge of the grid. Taking the start of the
    plateau avoids flagging more applications for a gain inside the noise.
    """
    best = f1_by_c.max()
    return float(f1_by_c[f1_by_c >= (1 - tolerance) * best].index.min())


def check_matches_sklearn(dev, settings, contamination, val_month=FINAL_VAL_MONTH):
    """Confirm threshold_for_rate gives the same alarms as sklearn's own
    `contamination` argument (fraction of validation rows that disagree)."""
    X_tr, _, X_va, _ = prepare_fold(dev, val_month, settings["feature_set"])
    kw = dict(n_estimators=int(settings["n_estimators"]),
              max_samples=int(settings["max_samples"]),
              max_features=float(settings["max_features"]))
    sk = fit_isolation_forest(X_tr, contamination=contamination, **kw)
    sk_flag = sk.predict(X_va) == -1
    manual = fit_isolation_forest(X_tr, **kw)
    mine = anomaly_score(manual, X_va) >= threshold_for_rate(
        anomaly_score(manual, X_tr), contamination)
    return float((sk_flag != mine).mean())


def compare_models(dev, dev_raw, settings, contamination, val_months=(3, 4, 5)):
    """Fit the Isolation Forest and the team logistic regression on every fold.

    `dev` has missing codes as NaN (forest); `dev_raw` keeps them (team model).
    Returns (metrics table, scores) where scores[val_month][model] holds
    (y, score) so figures and overlap statistics can reuse them.
    """
    rows, scores = [], {}
    for val_month in val_months:
        scores[val_month] = {}
        X_tr, _, X_va, y_va = prepare_fold(dev, val_month, settings["feature_set"])

        model = fit_isolation_forest(X_tr, int(settings["n_estimators"]),
                                     int(settings["max_samples"]),
                                     float(settings["max_features"]))
        s_tr, s_va = anomaly_score(model, X_tr), anomaly_score(model, X_va)
        m = evaluate(y_va, s_va)
        m.update(flag_metrics(y_va, s_va, threshold_for_rate(s_tr, contamination)))
        alert_rate = m["flag_rate"]      # same alert budget for the team model
        rows.append({"val_month": val_month, "model": "Isolation Forest", **m})
        scores[val_month]["Isolation Forest"] = (y_va, s_va)

        y_team, s_team = fit_team_model(dev_raw, val_month)
        if not np.array_equal(y_team, y_va):
            raise ValueError("forest and team model must be scored on the same rows")
        m = evaluate(y_va, s_team)
        m.update(flag_metrics(y_va, s_team, np.quantile(s_team, 1 - alert_rate)))
        rows.append({"val_month": val_month, "model": TEAM_MODEL, **m})
        scores[val_month][TEAM_MODEL] = (y_va, s_team)
    return pd.DataFrame(rows), scores


def seed_spread(dev, settings, val_month=FINAL_VAL_MONTH, seeds=SEEDS_FOR_SPREAD):
    """Re-fit the forest with several random seeds; return metrics per seed.

    Isolation Forest is random, so one seed can flatter or hurt it.
    """
    X_tr, _, X_va, y_va = prepare_fold(dev, val_month, settings["feature_set"])
    rows = []
    for seed in seeds:
        model = fit_isolation_forest(X_tr, int(settings["n_estimators"]),
                                     int(settings["max_samples"]),
                                     float(settings["max_features"]), seed=seed)
        m = evaluate(y_va, anomaly_score(model, X_va))
        rows.append({"seed": seed, "pr_auc": m["pr_auc"],
                     "roc_auc": m["roc_auc"], "recall_at_5fpr": m["recall_at_5fpr"]})
    return pd.DataFrame(rows)


def ks_selected_forest(dev, settings, k_list=(4, 8), val_months=(3, 4, 5)):
    """Diagnostic: does the forest improve if it only sees the best features?

    Per fold, rank the numeric features by KS (fraud vs legitimate) on the
    TRAINING months only and fit the forest on the top k plus their
    missing-indicators. This uses fraud labels to choose features, so it is
    semi-supervised, not a pure anomaly detector: it tests whether the weak
    result is caused by diluting the signal with 40+ uninformative columns.
    """
    numeric, _, _ = feature_columns(dev)
    rows = []
    for val_month in val_months:
        train, val = split_by_month(dev, range(val_month), val_month)
        ranked = eda.numeric_separation_table(train, numeric)["feature"].tolist()
        for k in k_list:
            cols = ranked[:k]
            prep = ColumnTransformer([("num", SimpleImputer(
                strategy="median", add_indicator=True), cols)])
            X_tr = prep.fit_transform(train).astype("float32")
            X_va = prep.transform(val).astype("float32")
            model = fit_isolation_forest(X_tr, int(settings["n_estimators"]),
                                         int(settings["max_samples"]), 1.0)
            m = evaluate(val[eda.TARGET].to_numpy(), anomaly_score(model, X_va))
            rows.append({"val_month": val_month, "top_k_features": k,
                         "features": ", ".join(cols), "pr_auc": m["pr_auc"],
                         "roc_auc": m["roc_auc"],
                         "recall_at_5fpr": m["recall_at_5fpr"]})
    return pd.DataFrame(rows)


def overlap_table(scores_by_model, fpr=eda.TARGET_FPR):
    """Which frauds does each model catch at 5% FPR, and how much do they differ?

    Returns (table, spearman) comparing the Isolation Forest with the team
    logistic regression: fraud caught by both, only the forest, only the
    team model, neither.
    """
    y, s_if = scores_by_model["Isolation Forest"]
    _, s_gb = scores_by_model[TEAM_MODEL]
    flag_if = s_if >= recall_at_fpr(y, s_if, fpr)["threshold_at_5fpr"]
    flag_gb = s_gb >= recall_at_fpr(y, s_gb, fpr)["threshold_at_5fpr"]
    fraud = y == 1
    n = int(fraud.sum())
    parts = {"caught by both": flag_if & flag_gb & fraud,
             "only Isolation Forest": flag_if & ~flag_gb & fraud,
             "only team logistic regression": ~flag_if & flag_gb & fraud,
             "caught by neither": ~flag_if & ~flag_gb & fraud}
    table = pd.DataFrame({"frauds": {k: int(v.sum()) for k, v in parts.items()}})
    table["share_of_frauds_pct"] = 100 * table["frauds"] / n
    rho_all = stats.spearmanr(s_if, s_gb)[0]
    rho_fraud = stats.spearmanr(s_if[fraud], s_gb[fraud])[0]
    return table, {"spearman_all": rho_all, "spearman_fraud_only": rho_fraud}


# ===========================================================================
# 6. Figures
# ===========================================================================

def plot_pr_curves(scores_by_model):
    """Precision-recall curves of all models on the held-back validation month."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    y = next(iter(scores_by_model.values()))[0]
    for name, (_, s) in scores_by_model.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{name}  (PR-AUC {average_precision_score(y, s):.3f})",
                zorder=3, **MODEL_STYLE[name])
    # No-skill line drawn underneath so the forest curve (which almost sits on
    # it) stays visible; the inset zooms into the bottom of the plot.
    ax.axhline(y.mean(), color=eda.NEUTRAL, lw=0.8, ls="-.", zorder=2,
               label=f"No-skill ({y.mean():.1%} fraud)")
    inset = ax.inset_axes([0.47, 0.30, 0.48, 0.30])
    for name, (_, s) in scores_by_model.items():
        p, r, _ = precision_recall_curve(y, s)
        inset.plot(r, p, **MODEL_STYLE[name])
    inset.axhline(y.mean(), color=eda.NEUTRAL, lw=0.8, ls="-.")
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 0.05)
    inset.set_yticks([0, 0.025, 0.05])
    inset.tick_params(labelsize=6)
    inset.set_title("Zoom: precision 0-0.05", fontsize=6.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.3, ax.get_ylim()[1]))
    ax.set_title("Precision-recall: unsupervised forest vs team logistic regression\n"
                 f"(trained on months 0-{FINAL_VAL_MONTH - 1}, scored on month {FINAL_VAL_MONTH})")
    ax.legend(loc="upper right")
    return fig


def plot_score_and_roc(scores_by_model):
    """Left: forest score by class with the 5% FPR cut-off. Right: recall vs FPR."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 3.0))
    y, s = scores_by_model["Isolation Forest"]
    thr = recall_at_fpr(y, s)["threshold_at_5fpr"]
    bins = np.linspace(np.quantile(s, 0.001), np.quantile(s, 0.999), 50)
    for cls in (0, 1):
        a1.hist(s[y == cls], bins=bins, density=True, histtype="stepfilled" if cls == 0 else "step",
                color=eda.COLOURS[cls], alpha=0.55 if cls == 0 else 1.0,
                lw=1.4, label=eda.LABELS[cls])
    a1.axvline(thr, color=eda.NEUTRAL, ls="--", lw=1)
    a1.text(thr, a1.get_ylim()[1] * 0.45, " 5% FPR\n cut-off", fontsize=7, color=eda.NEUTRAL)
    a1.set_xlabel("Anomaly score (higher = more unusual)")
    a1.set_ylabel("Density (per class)")
    a1.set_yticks([])
    a1.legend(loc="upper right")
    a1.set_title("Fraud scores only slightly higher")
    for name, (yy, ss) in scores_by_model.items():
        f, t, _ = roc_curve(yy, ss)
        a2.plot(f, t, label=name.replace(" (team repo)", ""), **MODEL_STYLE[name])
    a2.axvline(eda.TARGET_FPR, color=eda.NEUTRAL, ls="--", lw=1)
    a2.set_xlim(0, 0.2)
    a2.set_ylim(0, 0.8)
    a2.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    a2.set_xticklabels(["0%", "5%", "10%", "15%", "20%"])
    a2.set_xlabel("False-positive rate")
    a2.set_ylabel("Recall")
    a2.set_title("Recall at low FPR")
    a2.legend(loc="center right")
    fig.tight_layout()
    return fig


def plot_contamination(sweep, chosen):
    """Mean precision / recall / F1 of the forest versus `contamination`;
    `chosen` is the value picked by choose_contamination."""
    mean = sweep.groupby("contamination")[["precision", "recall", "f1"]].mean()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    styles = {"precision": ("#08306B", "--", "Precision"), "recall": ("#7F7F7F", ":", "Recall"),
              "f1": ("#B2182B", "-", "F1")}
    for col, (colour, ls, label) in styles.items():
        ax.plot(mean.index, mean[col], color=colour, ls=ls, lw=1.6, marker="o", ms=3.5, label=label)
    ax.axvline(chosen, color=eda.NEUTRAL, lw=0.8, ls="-.")
    ax.text(chosen * 0.97, ax.get_ylim()[1] * 0.9, f"chosen: {chosen:g}\n(F1 within 5% of best)",
            fontsize=7, color=eda.NEUTRAL, va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xticks(mean.index)
    ax.set_xticklabels([f"{c:g}" for c in mean.index])
    ax.minorticks_off()
    ax.set_xlabel("Contamination (share of applications flagged, log scale)")
    ax.set_ylabel("Mean over validation months 3-4")
    ax.set_title("Contamination moves the cut-off, not the ranking")
    ax.legend(loc="upper left")
    return fig
