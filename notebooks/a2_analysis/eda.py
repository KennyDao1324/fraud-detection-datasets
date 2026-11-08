"""
eda.py - Exploratory data analysis helpers for the BAF Base dataset.

Class balance, fraud rates, numeric distributions, correlations and drift.
Author: Thong. Reviewer: Kha.

Structure of this module
  1. Constants          - datasheet facts, colours, sizes, thresholds
  2. Loading / cleaning - load_base, missing_mask, with_missing_as_nan
  3. Statistics         - Wilson CI, fraud-rate tables, KS, Cramer's V,
                          correlations, PSI drift, missing patterns,
                          single-feature recall at 5% FPR
  4. Figures            - one plot_* function per report figure; each
                          returns a matplotlib Figure sized for A4
  5. Export             - save_figure (with width check), export_chart_data
                          (JSON for the A3 React dashboard)

The notebook notebooks/eda_A2-09.ipynb calls these functions in order.
Nothing here is random, so results are fully deterministic.
"""

import json
import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ===========================================================================
# 1. Constants
# ===========================================================================

TARGET = "fraud_bool"

# Columns where the value -1 means "missing" (BAF datasheet convention).
MISSING_MINUS_ONE = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
# intended_balcon_amount: any negative value means "missing".
MISSING_NEGATIVE = ["intended_balcon_amount"]

CATEGORICAL = ["payment_type", "employment_status", "housing_status",
               "source", "device_os"]

BINARY_FLAGS = ["email_is_free", "phone_home_valid", "phone_mobile_valid",
                "has_other_cards", "foreign_request", "keep_alive_session"]

# Months 0-5 = development (train/validation), 6-7 = test. Everything that
# could shape modelling choices uses months 0-5 only (risk R-A2-02).
DEV_MONTHS = range(0, 6)
SPLIT_MONTH = 5.5                     # boundary drawn between month 5 and 6

# Fairness split on customer_age (decade bins 10, 20, ..., 90).
# Both historical definitions are reported for comparison. The current
# production fairness audit uses >=50 (50+), not >50 (60+).
AGE_SPLIT = 50

# Analysis thresholds (named so the notebook text can refer to them).
MIN_LEVEL_N = 2000        # levels smaller than this are flagged as unreliable
CORR_LABEL_MIN = 0.15     # |rho| printed on the heatmap
CORR_PAIR_MIN = 0.40      # |rho| listed as a "strong pair"
SKEW_RATIO = 20           # 99th pct / median above this -> log10 axis
PSI_SHIFT = 0.25          # usual rule of thumb for a significant shift
TARGET_FPR = 0.05         # benchmark operating point (recall at 5% FPR)

# A4 page text width in inches (2.5 cm margins) and output resolution.
A4_WIDTH = 6.3
DPI = 300

# Colours. Legitimate/Fraud differ in lightness as well as hue (L* 67 vs 38)
# so they stay distinct in greyscale print and for colour-blind readers.
COLOURS = {0: "#74A9CF", 1: "#B2182B"}
RATE_COLOUR = COLOURS[1]      # any "fraud rate" bar or line
VOLUME_COLOUR = "#BDBDBD"     # application counts / shares
NEUTRAL = "#4D4D4D"           # reference lines and notes
LABELS = {0: "Legitimate", 1: "Fraud"}

# Units for the x-axis of each continuous feature (BAF datasheet).
X_UNITS = {
    "current_address_months_count": "months",
    "prev_address_months_count": "months",
    "bank_months_count": "months",
    "credit_risk_score": "score",
    "name_email_similarity": "similarity (0-1)",
    "bank_branch_count_8w": "applications",
    "intended_balcon_amount": "amount (non-missing only)",
    "days_since_request": "days",
    "velocity_6h": "applications / hour",
    "velocity_24h": "applications / hour",
    "velocity_4w": "applications / hour",
    "session_length_in_minutes": "minutes",
    "zip_count_4w": "applications",
    "date_of_birth_distinct_emails_4w": "emails",
}


# ===========================================================================
# 2. Loading and cleaning
# ===========================================================================

def set_style():
    """Apply one consistent, print-friendly plot style (all text >= 7 pt)."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": DPI,
        "font.size": 8,
        "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "figure.titlesize": 9.5, "figure.titleweight": "bold",
        "hatch.linewidth": 0.6,
    })


def load_base(path):
    """Read Base.csv (or the Kaggle Base.csv.zip directly).

    Numeric columns are downcast to save memory (risk R-A2-03); the
    statistics below cast back to float64 where precision matters.
    """
    df = pd.read_csv(path)
    for col in df.select_dtypes("float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes("int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def missing_mask(s):
    """Boolean Series: True where s holds the datasheet's missing code."""
    if s.name in MISSING_NEGATIVE:
        return s < 0
    if s.name in MISSING_MINUS_ONE:
        return s == -1
    return s.isna()


def with_missing_as_nan(df, columns=None):
    """Copy of the numeric columns with missing codes turned into NaN.

    Used for distributions and correlations so the -1 spike does not distort
    them. Missingness itself is analysed separately (Figure 3, pattern table).
    """
    columns = list(columns) if columns is not None else list(df.columns)
    out = df[columns].copy()
    for col in set(columns) & set(MISSING_MINUS_ONE + MISSING_NEGATIVE):
        out[col] = out[col].where(~missing_mask(out[col]))
    return out


def numeric_features(df):
    """Non-constant numeric feature columns (not target, month or 0/1 flags)."""
    skip = {TARGET, "month", *BINARY_FLAGS}
    return [c for c in df.select_dtypes("number").columns
            if c not in skip and df[c].nunique() > 1]


# ===========================================================================
# 3. Statistics
# ===========================================================================

def wilson_ci(successes, n, z=1.96):
    """95% Wilson confidence interval for a proportion (vectorised)."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def fraud_rate_table(df, by):
    """Applications, frauds, fraud rate (%) and 95% Wilson CI per group.

    `by` is a column name or a Series aligned with df (e.g. binned values).
    """
    g = df.groupby(by, observed=True)[TARGET].agg(n="size", frauds="sum")
    lo, hi = wilson_ci(g["frauds"], g["n"])
    g["rate"] = g["frauds"] / g["n"] * 100
    g["ci_low"] = lo * 100
    g["ci_high"] = hi * 100
    g["share"] = g["n"] / g["n"].sum() * 100
    return g


def profile_table(df):
    """One row per column: dtype, unique values, missing-code share, range.

    n_unique for float columns is counted after the float32 downcast in
    load_base, so it slightly understates the raw CSV.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        s64 = s.astype("float64") if is_num else s
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_unique": s64.nunique(),
            "missing_share_pct": round(float(missing_mask(s).mean()) * 100, 2),
            "min": round(float(s64.min()), 4) if is_num else None,
            "median": round(float(s64.median()), 4) if is_num else None,
            "max": round(float(s64.max()), 4) if is_num else None,
        })
    return pd.DataFrame(rows)


def numeric_separation_table(df, features):
    """Rank continuous features by fraud vs legitimate separation.

    KS statistic: 0 = identical distributions, 1 = fully separated.
    Missing codes are removed first, so the "missing_pct" column is reported
    alongside (KS is blind to the missingness signal, see Figure 3).
    """
    clean = with_missing_as_nan(df, features).astype("float64")
    y = df[TARGET].to_numpy()
    rows = []
    for col in features:
        fraud = clean.loc[y == 1, col].dropna()
        legit = clean.loc[y == 0, col].dropna()
        rows.append({"feature": col,
                     "missing_pct": round(clean[col].isna().mean() * 100, 1),
                     "median_legit": legit.median(),
                     "median_fraud": fraud.median(),
                     "ks_statistic": stats.ks_2samp(fraud, legit).statistic})
    return (pd.DataFrame(rows)
            .sort_values("ks_statistic", ascending=False)
            .reset_index(drop=True))


def cramers_v(x, y):
    """Bias-corrected Cramer's V between two categorical series."""
    table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(k_corr - 1, r_corr - 1))))


def _binned(s, n_bins=10):
    """Decile bins as strings, with the missing code kept as its own bin."""
    miss = missing_mask(s)
    if s.nunique() <= n_bins:
        b = s.astype(str)
    else:
        b = pd.qcut(s.where(~miss), n_bins, duplicates="drop").astype(str)
    return b.where(~miss, "missing")


def association_table(df, num_features, n_bins=10):
    """One comparable metric for every feature: bias-corrected Cramer's V.

    Numeric features are binned into deciles with "missing" as its own bin,
    so non-linear and missingness signal both count.
    """
    rows = []
    for col in CATEGORICAL:
        rows.append({"feature": col, "kind": "Categorical",
                     "V": cramers_v(df[col], df[TARGET])})
    for col in BINARY_FLAGS:
        rows.append({"feature": col, "kind": "Binary flag",
                     "V": cramers_v(df[col], df[TARGET])})
    for col in num_features:
        rows.append({"feature": col, "kind": "Numeric (deciles or levels)",
                     "V": cramers_v(_binned(df[col], n_bins), df[TARGET])})
    return (pd.DataFrame(rows).sort_values("V", ascending=False)
            .reset_index(drop=True))


def single_feature_recall(df, features, fpr=TARGET_FPR, n_bins=100):
    """Recall (TPR) at a fixed FPR when each feature is used ALONE as a score.

    Uses the benchmark metric itself, so it is comparable with the models in
    A2-11. Each level (categorical/flag) or quantile bin (numeric: up to
    `n_bins`, "missing" as its own bin) becomes one score value = its fraud
    rate, so both directions and non-linear shapes count. Bins are ranked
    from riskiest down and the ROC curve is interpolated at `fpr` (random
    tie-breaking). Fine bins are needed because the 5%-FPR cut falls inside
    the top decile; scores are fitted in-sample.
    """
    rows = []
    for col in features:
        bins = (df[col].astype(str) if col in CATEGORICAL + BINARY_FLAGS
                else _binned(df[col], n_bins))
        g = (df.groupby(bins, observed=True)[TARGET]
             .agg(frauds="sum", n="size").assign(legit=lambda t: t["n"] - t["frauds"]))
        g = g.assign(rate=g["frauds"] / g["n"]).sort_values("rate", ascending=False)
        fpr_curve = np.r_[0, g["legit"].cumsum() / g["legit"].sum()]
        tpr_curve = np.r_[0, g["frauds"].cumsum() / g["frauds"].sum()]
        rows.append({"feature": col,
                     "recall_at_5pct_fpr": float(np.interp(fpr, fpr_curve, tpr_curve))})
    return (pd.DataFrame(rows).sort_values("recall_at_5pct_fpr", ascending=False)
            .reset_index(drop=True))


def spearman_matrix(df, features):
    """Spearman correlation of numeric features (missing codes removed)."""
    return with_missing_as_nan(df, features).astype("float64").corr(method="spearman")


def correlated_pairs(corr, threshold=CORR_PAIR_MIN):
    """Feature pairs with |rho| >= threshold, strongest first."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = (upper.stack().dropna().rename("rho").reset_index()
             .rename(columns={"level_0": "feature_a", "level_1": "feature_b"}))
    pairs = pairs[pairs["rho"].abs() >= threshold]
    return pairs.sort_values("rho", key=abs, ascending=False).reset_index(drop=True)


def psi(ref_counts, cur_counts, eps=1e-4):
    """Population Stability Index between two binned distributions."""
    ref = np.clip(ref_counts / ref_counts.sum(), eps, None)
    cur = np.clip(cur_counts / cur_counts.sum(), eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_table(df, features, ref_month=0, n_bins=10):
    """PSI of each feature in every development month against `ref_month`.

    Bin edges come from the reference month (deciles); missing codes and
    category levels are their own bins.
    """
    rows = []
    months = sorted(df["month"].unique())
    for col in features:
        s = df[col]
        if col in CATEGORICAL + BINARY_FLAGS or s.nunique() <= n_bins:
            bins = s.astype(str)
        else:
            miss = missing_mask(s)
            ref = s[(df["month"] == ref_month) & ~miss]
            edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            bins = pd.cut(s.where(~miss), edges).astype(str).where(~miss, "missing")
        tab = pd.crosstab(bins, df["month"]).astype(float)
        row = {"feature": col}
        for m in months:
            if m != ref_month:
                row[f"month_{m}"] = psi(tab[ref_month].to_numpy(), tab[m].to_numpy())
        rows.append(row)
    out = pd.DataFrame(rows).set_index("feature")
    out["max_psi"] = out.max(axis=1)
    return out.sort_values("max_psi", ascending=False)


def missing_pattern_table(df, columns=("prev_address_months_count",
                                       "intended_balcon_amount",
                                       "bank_months_count")):
    """Fraud rate for every combination of missing / present in `columns`."""
    pattern = pd.Series("", index=df.index)
    for col in columns:
        pattern = pattern + np.where(missing_mask(df[col]), "M", "-")
    table = fraud_rate_table(df, pattern.rename("pattern"))
    short = {"prev_address_months_count": "prev_addr",
             "intended_balcon_amount": "balcon",
             "bank_months_count": "bank_months"}
    table.index = [" / ".join(f"{short.get(c, c)} {'MISSING' if p == 'M' else 'ok'}"
                              for c, p in zip(columns, key)) for key in table.index]
    table.index.name = "pattern"
    return table.sort_values("rate")


def age_split_table(df, split=AGE_SPLIT):
    """Share and fraud rate for both readings of the age fairness split."""
    rows = []
    for name, older in [(f"age > {split}", df["customer_age"] > split),
                        (f"age >= {split}", df["customer_age"] >= split)]:
        for label, mask in [("older", older), ("younger", ~older)]:
            rows.append({"definition": name, "group": label,
                         "applications": int(mask.sum()),
                         "share_pct": mask.mean() * 100,
                         "fraud_rate_pct": df.loc[mask, TARGET].mean() * 100})
    out = pd.DataFrame(rows)
    rates = out.pivot(index="definition", columns="group", values="fraud_rate_pct")
    out["rate_ratio_older_vs_younger"] = out["definition"].map(
        rates["older"] / rates["younger"])
    return out


# ===========================================================================
# 4. Figures (every function returns a Figure; text >= 7 pt at A4)
# ===========================================================================

def _note(ax_or_fig, text, y=-0.02):
    """Small grey footnote under a figure."""
    fig = ax_or_fig if isinstance(ax_or_fig, plt.Figure) else ax_or_fig.figure
    fig.text(0.01, y, "\n".join(textwrap.wrap(text, 120)), fontsize=7,
             color=NEUTRAL, ha="left", va="top")


def plot_class_balance(df):
    """Figure 1 - number and share of fraud vs legitimate applications."""
    counts = df[TARGET].value_counts().sort_index()
    share = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(A4_WIDTH, 1.6))
    bars = ax.barh([LABELS[i] for i in counts.index], counts.values,
                   color=[COLOURS[i] for i in counts.index], height=0.6)
    for bar, n, p in zip(bars, counts.values, share.values):
        ax.text(bar.get_width() + counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{n:,} ({p:.2%})", va="center", fontsize=7.5)
    ax.set_xlim(0, counts.max() * 1.3)            # linear: bar length = count
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e3:,.0f}k")
    ax.set_xlabel("Number of applications")
    ax.set_title(f"About 1 in {1 / share[1]:.0f} applications is fraud "
                 f"(all months 0-7, n = {counts.sum():,})")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_fraud_by_month(df):
    """Figure 2 - fraud rate (top) and volume (bottom) per month.

    Test months are greyed: shown for description only, no modelling or
    threshold decision uses them.
    """
    g = fraud_rate_table(df, "month")
    test = g.index > SPLIT_MONTH
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(A4_WIDTH, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
    for part, colour in [(~test, RATE_COLOUR), (test, VOLUME_COLOUR)]:
        sub = g[part]
        ax1.errorbar(sub.index, sub["rate"],
                     yerr=[sub["rate"] - sub["ci_low"], sub["ci_high"] - sub["rate"]],
                     color=colour, marker="o", markersize=4, capsize=3)
    ax1.set_ylim(0, g["ci_high"].max() * 1.2)
    ax1.set_ylabel("Fraud rate (%)\n(95% CI)")
    ax2.bar(g.index, g["n"] / 1000, width=0.7,
            color=[VOLUME_COLOUR if t else "#8C8C8C" for t in test])
    ax2.set_ylabel("Applications\n(thousands)")
    ax2.set_ylim(0, g["n"].max() / 1000 * 1.15)
    ax2.set_xlabel("Month")
    ax2.set_xticks(g.index)
    for ax in (ax1, ax2):
        ax.axvline(SPLIT_MONTH, color=NEUTRAL, linestyle="--", linewidth=0.9)
        ax.grid(axis="x", visible=False)
    top = ax1.get_ylim()[1]
    ax1.text(float(np.mean(g.index[~test])), top * 0.97, "Development (months 0-5)",
             ha="center", va="top", fontsize=7.5)
    ax1.text(float(np.mean(g.index[test])), top * 0.97, "Test (descriptive)",
             ha="center", va="top", fontsize=7.5, color=NEUTRAL)
    ax1.set_title("Fraud rate dips in months 1-3, then rises; volume falls after month 3")
    fig.align_ylabels([ax1, ax2])
    return fig


def plot_missing_codes(df, min_share=1.0):
    """Figure 3 - share of missing-coded values per column, split by class.

    Columns missing in under `min_share` % of rows are listed in a footnote.
    """
    cols = MISSING_MINUS_ONE + MISSING_NEGATIVE
    wide = pd.DataFrame({cls: [missing_mask(df.loc[df[TARGET] == cls, c]).mean() * 100
                               for c in cols] for cls in (0, 1)}, index=cols)
    rare = wide[wide.max(axis=1) < min_share]
    wide = wide[wide.max(axis=1) >= min_share].sort_values(0)
    fig, ax = plt.subplots(figsize=(A4_WIDTH, 0.55 * len(wide) + 0.9))
    y = np.arange(len(wide))
    h = 0.38
    for k, cls in enumerate((0, 1)):
        bars = ax.barh(y + (0.5 - k) * h, wide[cls], h, color=COLOURS[cls],
                       hatch="////" if cls == 1 else None, edgecolor="white",
                       label=LABELS[cls])
        ax.bar_label(bars, fmt="%.1f%%", fontsize=7, padding=2)
    ax.set_yticks(y, wide.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Applications with the missing-value code (%)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Missing codes are more frequent in fraud applications (months 0-5)")
    fig.tight_layout()
    if len(rare):
        _note(fig, f"Not shown (< {min_share:g}% missing in both classes): "
              + ", ".join(rare.index))
    return fig


def plot_fraud_rate_by_category(df, min_n=MIN_LEVEL_N):
    """Figure 4 - fraud rate per level of each categorical feature (shared
    y-axis), with 95% Wilson CIs and the overall rate as a reference."""
    overall = df[TARGET].mean() * 100
    ncols = 3
    nrows = int(np.ceil((len(CATEGORICAL) + 1) / ncols))   # +1 panel for the key
    fig, axes = plt.subplots(nrows, ncols, figsize=(A4_WIDTH, 2.3 * nrows),
                             sharey=True)
    axes = axes.ravel()
    for ax, col in zip(axes, CATEGORICAL):
        g = fraud_rate_table(df, col).sort_values("n", ascending=False)
        small = (g["n"] < min_n).to_numpy()
        bars = ax.bar(g.index.astype(str), g["rate"],
                      color=["white" if s else RATE_COLOUR for s in small],
                      edgecolor=RATE_COLOUR, linewidth=0.8,
                      yerr=[g["rate"] - g["ci_low"], g["ci_high"] - g["rate"]],
                      capsize=2, error_kw={"linewidth": 0.8, "ecolor": NEUTRAL})
        for bar, s in zip(bars, small):
            if s:
                bar.set_hatch("////")
        ax.axhline(overall, color=NEUTRAL, linestyle="--", linewidth=0.8)
        ax.set_title(col, fontsize=8.5)
        ax.grid(axis="x", visible=False)
        rotate = len(g) > 2
        plt.setp(ax.get_xticklabels(), rotation=45 if rotate else 0,
                 ha="right" if rotate else "center", rotation_mode="anchor")
    for ax in axes[::ncols]:
        ax.set_ylabel("Fraud rate (%)")
    key = axes[len(CATEGORICAL)]
    key.axis("off")
    key.text(0.02, 0.5,
             f"Dashed line = overall rate,\nmonths 0-5 ({overall:.2f}%)\n"
             "Whiskers = 95% Wilson CI\nLevels sorted by size\n"
             f"Hatched = fewer than {min_n:,}\napplications (unreliable)\n"
             "Level codes are anonymised\nin the dataset",
             fontsize=7.5, va="center", transform=key.transAxes)
    for ax in axes[len(CATEGORICAL) + 1:]:
        ax.axis("off")
    fig.suptitle("Fraud rate by level of each categorical feature (months 0-5)")
    fig.tight_layout()
    return fig


def plot_binary_flags(df):
    """Figure 5 - fraud rate when each yes/no flag is 0 vs 1."""
    overall = df[TARGET].mean() * 100
    tables = {col: fraud_rate_table(df, col) for col in BINARY_FLAGS}
    fig, ax = plt.subplots(figsize=(A4_WIDTH, 2.6))
    x = np.arange(len(BINARY_FLAGS))
    width = 0.38
    top = 0
    for i, (val, colour, lab) in enumerate([(0, "#BDBDBD", "No (0)"),
                                            (1, "#525252", "Yes (1)")]):
        rate = np.array([tables[c]["rate"].get(val, np.nan) for c in BINARY_FLAGS])
        lo = np.array([tables[c]["ci_low"].get(val, np.nan) for c in BINARY_FLAGS])
        hi = np.array([tables[c]["ci_high"].get(val, np.nan) for c in BINARY_FLAGS])
        ax.bar(x + (i - 0.5) * width, rate, width, color=colour, label=lab,
               yerr=[rate - lo, hi - rate], capsize=2, error_kw={"linewidth": 0.8})
        top = max(top, np.nanmax(hi))
    ax.axhline(overall, color=NEUTRAL, linestyle="--", linewidth=0.8,
               label=f"Overall, months 0-5 ({overall:.2f}%)")
    ax.set_xticks(x, BINARY_FLAGS, rotation=15, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_ylim(0, top * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title("Fraud rate when each binary flag is 0 or 1 (95% CI, months 0-5)")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df, features, ks=None):
    """Figure 6 - per-class density of the most separating continuous features.

    Values outside the pooled 1st-99th percentile are FILTERED OUT (not
    clipped: clipping piles the tail into the last bin and draws a fake
    spike). Missing codes are removed. Legitimate = filled, Fraud = outline.
    `ks` is an optional {feature: KS statistic} shown under each panel.
    """
    clean = with_missing_as_nan(df, features).astype("float64")
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(A4_WIDTH, 1.75 * nrows + 0.5))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, features):
        s = clean[col]
        lower, upper = s.quantile([0.01, 0.99])
        keep = s.between(lower, upper)
        median = s.median()
        log_axis = lower >= 0 and median > 0 and upper / median > SKEW_RATIO
        shift = 1 if log_axis and lower == 0 else 0     # log(x + 1) when zeros exist
        is_int = pd.api.types.is_integer_dtype(df[col])
        if log_axis:
            # Bin and normalise in log10 space: a linear-width density on
            # log-spaced bins would make the wide right-hand bins look empty.
            bins = np.linspace(np.log10(lower + shift), np.log10(upper + shift), 41)
            if is_int:   # merge bins narrower than one integer step
                edges = np.unique(np.round(10 ** bins) - 0.5)
                bins = np.log10(np.clip(edges, 0.5, None))
        elif is_int and upper > lower:
            width = max(1, int(np.ceil((upper - lower) / 40)))
            bins = np.arange(lower - 0.5, upper + width, width)   # integer-aligned
        else:
            bins = np.linspace(lower, upper, 41)                 # shared by both classes
        for cls in (0, 1):
            values = s[keep & (df[TARGET] == cls).to_numpy()].dropna() + shift
            if log_axis:
                values = np.log10(values)
            style = (dict(fill=True, alpha=0.55, linewidth=0) if cls == 0
                     else dict(fill=False, linewidth=1.1))
            sns.histplot(values, ax=ax, bins=bins, stat="density", element="step",
                         color=COLOURS[cls], label=LABELS[cls], **style)
        if log_axis:   # ticks at 1, 10, 100, ... in original units
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            ax.xaxis.set_major_formatter(lambda v, _: f"{10 ** v:g}")
        unit = X_UNITS.get(col, "") + (" + 1" if shift else "") + (", log scale" if log_axis else "")
        ax.set_xlabel(unit if ks is None else f"{unit}  |  KS = {ks[col]:.2f}",
                      fontsize=7)
        ax.set_title(col, fontsize=7.5)
        ax.set_ylabel("")
        ax.set_yticklabels([])           # density values carry no meaning here
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", visible=False)
    for ax in axes[::ncols]:
        ax.set_ylabel("Density (per class)", fontsize=7)
    for ax in axes[len(features):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Continuous features with the largest fraud vs legitimate difference\n"
                 "(months 0-5; 1st-99th percentile shown, missing codes excluded)",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    return fig


def plot_correlations(corr, threshold=CORR_LABEL_MIN):
    """Figure 7 - Spearman correlation heatmap (lower triangle).

    Takes the matrix from spearman_matrix(). Features with no |rho| >=
    threshold against any other feature are dropped and listed below.
    """
    off = corr.abs().where(~np.eye(len(corr), dtype=bool))
    keep = off.max() >= threshold
    dropped = list(corr.index[~keep])
    corr = corr.loc[keep, keep].iloc[1:, :-1]   # drop empty first row / last column
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    def fmt(v):
        return "" if abs(v) < threshold else f"{v:.2f}".replace("0.", ".").replace("-", "−")
    labels = corr.apply(lambda c: c.map(fmt))
    fig, ax = plt.subplots(figsize=(A4_WIDTH - 0.4, 0.33 * len(corr) + 1.9))
    sns.heatmap(corr, mask=mask, cmap="vlag", vmin=-1, vmax=1, center=0,
                annot=labels, fmt="", annot_kws={"fontsize": 7},
                linewidths=0.4, square=True,
                cbar_kws={"shrink": 0.6, "label": "Spearman ρ"}, ax=ax)
    ax.grid(False)
    ax.set_title(f"Spearman correlation between numeric features "
                 f"(|ρ| ≥ {threshold} printed, months 0-5)")
    ax.tick_params(axis="both", labelsize=7, length=0)
    fig.tight_layout()
    if dropped:
        _note(fig, f"Not shown (|ρ| < {threshold} with every other feature): "
              + ", ".join(dropped), y=0.0)
    return fig


def plot_target_association(assoc, top=None):
    """Figure 8 - every feature ranked by Cramer's V with fraud (one metric)."""
    data = assoc if top is None else assoc.head(top)
    palette = {"Categorical": "#525252", "Numeric (deciles or levels)": "#969696",
               "Binary flag": "#D9D9D9"}
    fig, ax = plt.subplots(figsize=(A4_WIDTH, 0.15 * len(data) + 0.9))
    y = np.arange(len(data))
    for kind, colour in palette.items():
        m = (data["kind"] == kind).to_numpy()
        ax.barh(y[m], data["V"].to_numpy()[m], color=colour, edgecolor="#525252",
                linewidth=0.5, label=kind, height=0.75)
    ax.set_yticks(y, data["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Cramér's V with fraud (bias-corrected; 0 = none, 1 = perfect)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", title="Feature type", title_fontsize=7.5)
    ax.set_title("Every feature is only weakly associated with fraud (months 0-5)")
    fig.tight_layout()
    return fig


def plot_fraud_by_age(df, split=AGE_SPLIT):
    """Figure 9 - fraud rate by age band; second tick line = share of rows.

    Two boundaries compare historical definitions: dashed = ">50" (60+),
    dotted = ">=50" (50+, used in the current production fairness audit).
    """
    g = fraud_rate_table(df, "customer_age").reset_index()
    fig, ax = plt.subplots(figsize=(A4_WIDTH, 2.7))
    x = np.arange(len(g))
    ax.bar(x, g["rate"], color=RATE_COLOUR, width=0.65,
           yerr=[g["rate"] - g["ci_low"], g["ci_high"] - g["rate"]], capsize=3,
           error_kw={"linewidth": 0.8, "ecolor": NEUTRAL})
    ax.set_xticks(x, [f"{a}\n{s:.1f}%" if s >= 0.1 else f"{a}\n<0.1%"
                      for a, s in zip(g["customer_age"], g["share"])])
    ax.set_xlabel("Age band in years (second line: share of applications)")
    ax.set_ylabel("Fraud rate (%)\n(95% CI)")
    ax.set_ylim(0, None)
    ages = g["customer_age"].to_numpy()
    gt = np.searchsorted(ages, split, side="right") - 0.5   # between 50 and 60
    ge = np.searchsorted(ages, split, side="left") - 0.5    # between 40 and 50
    top = ax.get_ylim()[1]
    ax.axvline(gt, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax.axvline(ge, color=NEUTRAL, linestyle=":", linewidth=0.9)
    ax.text(gt + 0.1, top * 0.95, f"> {split}\n(historical comparison)",
            ha="left", va="top", fontsize=7)
    ax.text(ge - 0.1, top * 0.95, f"≥ {split}\n(current audit)",
            ha="right", va="top", fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set_title("Fraud rate rises with every age band (months 0-5)")
    fig.tight_layout()
    return fig


def plot_ordinal_features(df, features=("income", "proposed_credit_limit",
                                         "credit_risk_score"), min_n=MIN_LEVEL_N):
    """Figure 10 - fraud rate across the values of ordinal features.

    Features with more than 20 values are binned into deciles (labelled by
    upper bound). Levels with fewer than `min_n` rows are dropped.
    """
    xlab = {"income": "Income (dataset value, 0.1-0.9)",
            "proposed_credit_limit": "Proposed credit limit",
            "credit_risk_score": "Score decile (upper bound)"}
    overall = df[TARGET].mean() * 100
    fig, axes = plt.subplots(1, len(features), figsize=(A4_WIDTH, 2.5))
    for ax, col in zip(np.atleast_1d(axes), features):
        values = df[col]
        binned = values.nunique() > 20
        if binned:
            values = pd.qcut(values, 10, duplicates="drop")
        g = fraud_rate_table(df, values)
        g = g[g["n"] >= min_n]
        x = np.arange(len(g))
        ax.errorbar(x, g["rate"], yerr=[g["rate"] - g["ci_low"], g["ci_high"] - g["rate"]],
                    color=RATE_COLOUR, marker="o", markersize=3, capsize=2, linewidth=1.2)
        ax.axhline(overall, color=NEUTRAL, linestyle="--", linewidth=0.8)
        labels = ([f"≤{iv.right:.0f}" for iv in g.index] if binned
                  else [f"{v:g}" for v in g.index])
        rotate = len(x) > 5
        ax.set_xticks(x, labels, rotation=60 if rotate else 0,
                      ha="right" if rotate else "center", rotation_mode="anchor",
                      fontsize=7)
        ax.set_xlabel(xlab.get(col, col), fontsize=7.5)
        ax.set_title(col, fontsize=8)
        ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Fraud rate (%)")
    fig.suptitle(f"Fraud rate is highest at the top of each scale "
                 f"(dashed = overall, {overall:.2f}%; months 0-5)", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_feature_drift(drift, threshold=PSI_SHIFT, top=10):
    """Figure 11 - PSI of the most-drifting features in months 1-5 vs month 0."""
    data = drift.head(top).drop(columns="max_psi")
    fig, ax = plt.subplots(figsize=(A4_WIDTH, 0.22 * len(data) + 1.2))
    sns.heatmap(data, cmap="Reds", vmin=0, vmax=max(1.0, threshold * 4),
                annot=data.apply(lambda c: c.map(lambda v: f"{v:.2f}")),
                fmt="", annot_kws={"fontsize": 7},
                linewidths=0.4, cbar_kws={"label": "PSI vs month 0 (capped colour)"},
                ax=ax)
    ax.set_xticklabels([c.replace("month_", "Month ") for c in data.columns],
                       rotation=0)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=7, length=0)
    ax.set_title(f"Feature drift in months 1-5 vs month 0 (PSI > {threshold} = large shift)")
    fig.tight_layout()
    return fig


# ===========================================================================
# 5. Export
# ===========================================================================

def save_figure(fig, name, folder, max_width=A4_WIDTH + 0.05):
    """Save a 300-dpi PNG, warn if it is wider than the A4 text width, close it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    width_in = plt.imread(path).shape[1] / DPI
    if width_in > max_width:
        warnings.warn(f"{name}: saved width {width_in:.2f} in > {A4_WIDTH} in; "
                      "fonts will shrink when the figure is fitted to A4")
    plt.close(fig)
    return path


def export_chart_data(tables, path):
    """Write the data behind the dashboard charts as one JSON file (for A3).

    `tables` maps a chart name to a DataFrame; the index is kept as a column.
    """
    out = {name: json.loads(t.reset_index().to_json(orient="records", double_precision=6))
           for name, t in tables.items()}
    out["_meta"] = {"months": {k: ("0-7" if k == "monthly" else "0-5") for k in tables},
                    "rates_in": "percent", "ci": "95% Wilson",
                    "source": "BAF Base (Jesus et al., 2022)"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1))
    return Path(path)
