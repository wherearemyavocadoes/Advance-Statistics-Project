#!/usr/bin/env python3
"""
============================================================
Advanced Statistics Final Project — OOP Healthcare Analysis
============================================================
Clean analysis script covering all 9 Research Questions.
Generates LaTeX-formatted regression tables and figures.
============================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, classification_report

# ── Style ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
BLUE, ORANGE, RED, GREEN, GREY = "#4C9BE8", "#E8714C", "#D94F3D", "#5BA85A", "#888888"

OUT = "oop_outputs"
os.makedirs(OUT, exist_ok=True)


# ══════════════════════════════════════════════════════════
# CELL 0 — DATA LOAD & DERIVED VARIABLES
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING AND PREPROCESSING DATA")
print("=" * 70)

df = pd.read_csv("preprocessed_data.csv", low_memory=False)
print(f"Data loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# Numeric coercion
num_cols = [
    "age", "household_size", "monthly_food_spend", "monthly_edu_spend",
    "op_oope_total", "ip_oope_total", "net_oope_op", "net_oope_ip",
    "net_oope_total", "total_hh_spend", "oope_share_income",
    "catastrophic_10", "district_uhc_index", "op_private", "ip_private",
    "had_op_visit", "had_ip_visit", "had_any_visit", "log_net_oope_total"
]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Core derived variables
df["is_insured"]  = (df["has_insurance"] == "Have Insurance").astype(int)
df["is_urban"]    = (df["rural_urban"] == "Urban").astype(int)
df["is_male"]     = (df["gender"] == "Male").astype(int)
df["total_oope"]  = df["net_oope_total"]
df["log_total_oope"] = np.log1p(df["total_oope"])

# Consumption / income proxy
df["monthly_consumption_proxy"] = df["total_hh_spend"]
df["annual_consumption_proxy"]  = df["monthly_consumption_proxy"] * 12
df["log_consumption"] = np.log1p(df["monthly_consumption_proxy"])
df["wealth_10k"] = df["monthly_consumption_proxy"] / 10000

# Catastrophic expenditure
df["catastrophic"] = df["catastrophic_10"]

# UHC standardisation
df["uhc_std"] = (
    df["district_uhc_index"] - df["district_uhc_index"].mean()
) / df["district_uhc_index"].std()

# Health burden proxy
df["poor_health"] = df["physical_health_rating"].isin(
    ["Poor", "Very Poor", "Average"]
).astype(int)

# Visit indicators
df["visited_op"] = (df["had_op_visit"] == 1).astype(int)
df["visited_ip"] = (df["had_ip_visit"] == 1).astype(int)

print("Derived variables created.\n")
print(df[["total_oope", "total_hh_spend", "catastrophic",
           "is_insured", "op_private", "ip_private",
           "district_uhc_index"]].describe().round(2))


# ══════════════════════════════════════════════════════════
# HELPER: LaTeX regression table
# ══════════════════════════════════════════════════════════
def results_to_latex(results_df, caption, label, filename):
    """Save a regression results DataFrame as a LaTeX table."""
    latex = results_df.to_latex(
        float_format="%.4f",
        caption=caption,
        label=label,
        column_format="l" + "r" * len(results_df.columns),
        escape=False,
    )
    path = os.path.join(OUT, filename)
    with open(path, "w") as f:
        f.write(latex)
    print(f"  → LaTeX table saved: {path}")
    return latex


# ══════════════════════════════════════════════════════════
# CELL 1 — RQ1 + RQ2: Insurance & OOP
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RQ1/RQ2: INSURANCE AND OOP EXPENDITURE")
print("=" * 70)

q1 = df[[
    "total_oope", "log_total_oope", "is_insured", "has_insurance",
    "age", "household_size", "is_urban", "is_male",
    "log_consumption", "poor_health", "visited_op", "visited_ip",
    "op_private", "ip_private", "district_uhc_index",
    "education", "occupation", "caste", "religion"
]].replace([np.inf, -np.inf], np.nan).dropna()

print("\nDescriptive statistics by insurance status:")
desc = q1.groupby("has_insurance")["total_oope"].agg(
    N="count", Mean="mean", Median="median", Std="std"
).round(2)
print(desc)

insured = q1.loc[q1["is_insured"] == 1, "total_oope"]
uninsured = q1.loc[q1["is_insured"] == 0, "total_oope"]

u_stat, p_val = stats.mannwhitneyu(insured, uninsured, alternative="two-sided")
print(f"\nMann-Whitney U test: U={u_stat:,.0f}, p={p_val:.4f}")

# Interaction terms
q1["ins_x_urban"]    = q1["is_insured"] * q1["is_urban"]
q1["ins_x_male"]     = q1["is_insured"] * q1["is_male"]
q1["ins_x_log_cons"] = q1["is_insured"] * q1["log_consumption"]

formula_q1 = """
log_total_oope ~ is_insured + log_consumption + age + household_size
+ is_urban + is_male + poor_health + visited_op + visited_ip
+ op_private + ip_private + district_uhc_index
+ ins_x_urban + ins_x_male + ins_x_log_cons
+ C(education) + C(occupation)
"""

model_q1 = smf.ols(formula_q1, data=q1).fit(cov_type="HC3")
print("\n" + model_q1.summary().as_text())

# Build results table
key_vars = [
    "is_insured", "log_consumption", "age", "household_size",
    "is_urban", "is_male", "poor_health", "visited_op", "visited_ip",
    "op_private", "ip_private", "district_uhc_index",
    "ins_x_urban", "ins_x_male", "ins_x_log_cons"
]
q1_results = pd.DataFrame({
    "Coefficient": model_q1.params[key_vars].round(4),
    "Std Error":   model_q1.bse[key_vars].round(4),
    "p-value":     model_q1.pvalues[key_vars].round(4),
    "% Effect":    ((np.exp(model_q1.params[key_vars]) - 1) * 100).round(4)
})
q1_results.to_csv(f"{OUT}/rq1_insurance_results.csv")
results_to_latex(q1_results,
    caption="RQ1/RQ2: OLS Regression — Insurance Effect on log(OOP)",
    label="tab:rq1_ols",
    filename="rq1_ols_table.tex")

# Save model summary
with open(f"{OUT}/rq1_model_summary.txt", "w") as f:
    f.write(model_q1.summary().as_text())

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
q1.groupby("has_insurance")["total_oope"].median().plot.bar(
    ax=axes[0], color=[BLUE, ORANGE], edgecolor="black")
axes[0].set_title("Median OOP by Insurance Status")
axes[0].set_ylabel("₹")
axes[0].tick_params(axis="x", rotation=0)

top_vars = q1_results.sort_values("p-value").head(8)
axes[1].barh(top_vars.index, top_vars["Coefficient"], color=BLUE, edgecolor="black")
axes[1].set_title("Top Predictors (by p-value)")
axes[1].set_xlabel("Coefficient")
plt.tight_layout()
plt.savefig(f"{OUT}/rq1_rq2_insurance.png", dpi=150)
plt.close()
print("  → Figure saved: rq1_rq2_insurance.png")


# ══════════════════════════════════════════════════════════
# CELL 2 — RQ3: Public vs Private
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RQ3: PUBLIC vs PRIVATE HEALTHCARE COST")
print("=" * 70)

q3 = df[["total_oope", "log_total_oope", "op_private", "ip_private",
          "is_insured", "age", "is_urban", "is_male",
          "household_size", "log_consumption", "poor_health",
          "district_uhc_index"]].replace([np.inf, -np.inf], np.nan).dropna()

formula_q3 = """
log_total_oope ~ op_private + ip_private + is_insured + age
+ is_urban + is_male + household_size + log_consumption
+ poor_health + district_uhc_index
"""
model_q3 = smf.ols(formula_q3, data=q3).fit(cov_type="HC3")
print(model_q3.summary().as_text())

q3_vars = ["op_private", "ip_private", "is_insured", "age",
            "is_urban", "is_male", "household_size",
            "log_consumption", "poor_health", "district_uhc_index"]
q3_results = pd.DataFrame({
    "Coefficient": model_q3.params[q3_vars].round(4),
    "Std Error":   model_q3.bse[q3_vars].round(4),
    "p-value":     model_q3.pvalues[q3_vars].round(4),
    "% Effect":    ((np.exp(model_q3.params[q3_vars]) - 1) * 100).round(4)
})
q3_results.to_csv(f"{OUT}/rq3_public_private_results.csv")
results_to_latex(q3_results,
    caption="RQ3: OLS Regression — Public vs Private Facility Effect on log(OOP)",
    label="tab:rq3_ols",
    filename="rq3_ols_table.tex")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
means = q3.groupby(["op_private", "ip_private"])["total_oope"].median().reset_index()
means["Label"] = means.apply(
    lambda r: f"OP={'Pvt' if r['op_private'] else 'Pub'}, IP={'Pvt' if r['ip_private'] else 'Pub'}", axis=1)
ax.bar(means["Label"], means["total_oope"], color=[BLUE, ORANGE, GREEN, RED], edgecolor="black")
ax.set_ylabel("Median OOP (₹)")
ax.set_title("Median OOP by Public/Private Facility Usage")
plt.tight_layout()
plt.savefig(f"{OUT}/rq3_public_private.png", dpi=150)
plt.close()
print("  → Figure saved: rq3_public_private.png")


# ══════════════════════════════════════════════════════════
# CELL 3 — RQ4 + RQ5: Spline Threshold (FIXED CONVERGENCE)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RQ4/RQ5: INCOME THRESHOLDS — SPLINE LOGISTIC REGRESSION")
print("=" * 70)

q4 = df[["catastrophic", "total_hh_spend", "is_insured",
          "is_urban", "district_uhc_index"]].dropna()
q4["log_income"] = np.log1p(q4["total_hh_spend"])

# Knots at 25th, 50th, 75th percentiles
knots = np.percentile(q4["log_income"], [25, 50, 75])
for i, k in enumerate(knots, 1):
    q4[f"knot_{i}"] = np.maximum(0, q4["log_income"] - k)

# FIX: Increase maxiter and use different solver for convergence
formula_spline = """
catastrophic ~ log_income + knot_1 + knot_2 + knot_3
+ is_insured + is_urban + district_uhc_index
"""
model_spline = smf.logit(formula_spline, data=q4).fit(
    method="bfgs", maxiter=500, disp=False
)
print(model_spline.summary())
print(f"\nConverged: {model_spline.mle_retvals['converged']}")

# Odds ratios
or_spline = pd.DataFrame({
    "OR":      np.exp(model_spline.params).round(4),
    "p-value": model_spline.pvalues.round(4)
})
or_spline.to_csv(f"{OUT}/rq4_rq5_spline_odds.csv")
results_to_latex(or_spline,
    caption="RQ4/RQ5: Logistic Spline Regression — Odds Ratios for Catastrophic Expenditure",
    label="tab:rq4_spline",
    filename="rq4_rq5_spline_table.tex")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
grid = np.linspace(q4["log_income"].min(), q4["log_income"].max(), 300)
base = pd.DataFrame({
    "log_income": grid,
    "is_insured": 0, "is_urban": 0,
    "district_uhc_index": q4["district_uhc_index"].median()
})
for i, k in enumerate(knots, 1):
    base[f"knot_{i}"] = np.maximum(0, base["log_income"] - k)
pred = model_spline.predict(base)
axes[0].plot(np.expm1(grid), pred, color=BLUE, lw=2)
for k in knots:
    axes[0].axvline(np.expm1(k), ls="--", color=GREY, alpha=.6)
axes[0].set_xlabel("Monthly household spend (₹)")
axes[0].set_ylabel("P(Catastrophic)")
axes[0].set_title("Spline Probability Curve")

or_plot = or_spline.drop("Intercept")
colors = [RED if v > 1 else GREEN for v in or_plot["OR"]]
axes[1].barh(or_plot.index, or_plot["OR"], color=colors, edgecolor="black")
axes[1].axvline(1, color="black", lw=1)
axes[1].set_xlabel("Odds Ratio")
axes[1].set_title("Odds Ratios (Catastrophic)")
plt.tight_layout()
plt.savefig(f"{OUT}/rq4_rq5_spline_threshold.png", dpi=150)
plt.close()
print("  → Figure saved: rq4_rq5_spline_threshold.png")


# ══════════════════════════════════════════════════════════
# CELL 4 — RQ6 + RQ9: District Effects
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RQ6/RQ9: DISTRICT EFFECTS AND UHC INFRASTRUCTURE")
print("=" * 70)

q6 = df[["catastrophic", "wealth_10k", "age", "household_size",
          "is_insured", "is_urban", "op_private", "ip_private",
          "uhc_std", "district"]].dropna()

formula_q6 = """
catastrophic ~ wealth_10k + age + household_size
+ is_insured + is_urban + op_private + ip_private + uhc_std
"""
model_q6 = smf.logit(formula_q6, data=q6).fit(
    method="bfgs", maxiter=500, disp=False
)
print(model_q6.summary())

or_dist = pd.DataFrame({
    "OR":      np.exp(model_q6.params).round(4),
    "p-value": model_q6.pvalues.round(4)
})
or_dist.to_csv(f"{OUT}/rq6_rq9_district_odds.csv")
results_to_latex(or_dist,
    caption="RQ6/RQ9: Logistic Regression — District-Level Determinants of Catastrophic Expenditure",
    label="tab:rq6_logit",
    filename="rq6_rq9_district_table.tex")

# Mixed-effects model (random intercept by district)
print("\nFitting mixed-effects model (random intercept by district)...")
try:
    me_formula = "catastrophic ~ wealth_10k + age + household_size + is_insured + is_urban + op_private + ip_private + uhc_std"
    model_me = smf.mixedlm(me_formula, data=q6, groups=q6["district"]).fit(
        method="lbfgs", maxiter=500
    )
    print(model_me.summary())
    me_results = pd.DataFrame({
        "Coefficient": model_me.fe_params.round(4),
        "Std Error":   model_me.bse_fe.round(4),
        "p-value":     model_me.pvalues.round(4)
    })
    results_to_latex(me_results,
        caption="RQ9: Mixed-Effects Model — Random Intercept by District",
        label="tab:rq9_mixed",
        filename="rq9_mixed_table.tex")
except Exception as e:
    print(f"  Mixed-effects model warning: {e}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
agg = q6.groupby("district").agg(
    cat_rate=("catastrophic", "mean"),
    uhc=("uhc_std", "mean")
).dropna()
axes[0].scatter(agg["uhc"], agg["cat_rate"], alpha=.5, s=20, color=BLUE)
z = np.polyfit(agg["uhc"], agg["cat_rate"], 1)
axes[0].plot(sorted(agg["uhc"]), np.polyval(z, sorted(agg["uhc"])), color=RED, lw=2)
axes[0].set_xlabel("UHC Index (standardised)")
axes[0].set_ylabel("Catastrophic Rate")
axes[0].set_title("District UHC vs Catastrophic Rate")

or_plot = or_dist.drop("Intercept")
colors = [RED if v > 1 else GREEN for v in or_plot["OR"]]
axes[1].barh(or_plot.index, or_plot["OR"], color=colors, edgecolor="black")
axes[1].axvline(1, color="black", lw=1)
axes[1].set_xlabel("Odds Ratio")
axes[1].set_title("Logistic Odds Ratios")
plt.tight_layout()
plt.savefig(f"{OUT}/rq6_rq9_district_effects.png", dpi=150)
plt.close()
print("  → Figure saved: rq6_rq9_district_effects.png")


# ══════════════════════════════════════════════════════════
# CELL 5 — RQ7: LASSO Variable Selection
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RQ7: LASSO VARIABLE SELECTION")
print("=" * 70)

cat_features = ["education", "occupation", "caste", "religion",
                 "marital_status", "physical_health_rating",
                 "mental_health_rating", "state_region",
                 "district_uhc_tercile"]
num_features = ["age", "household_size", "is_insured", "is_urban",
                 "is_male", "total_hh_spend", "poor_health",
                 "had_op_visit", "had_ip_visit", "op_private",
                 "ip_private", "district_uhc_index",
                 "ins_premium_annual", "ins_coverage_limit"]
cat_features = [c for c in cat_features if c in df.columns]
num_features = [c for c in num_features if c in df.columns]

q7 = df[cat_features + num_features + ["log_total_oope"]].dropna()
X = q7[cat_features + num_features]
y = q7["log_total_oope"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist"),
     cat_features)
])
X_tr = pre.fit_transform(X)
feat_names = (num_features +
              list(pre.named_transformers_["cat"].get_feature_names_out(cat_features)))

from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_tr, y)
coefs = pd.Series(lasso.coef_, index=feat_names)
selected = coefs[coefs != 0].sort_values(key=abs, ascending=False)

lasso_df = pd.DataFrame({
    "feature": selected.index,
    "coef": selected.values.round(4),
    "abs_coef": np.abs(selected.values).round(4)
})
lasso_df.to_csv(f"{OUT}/rq7_lasso_selected_predictors.csv", index=False)
results_to_latex(lasso_df.set_index("feature"),
    caption="RQ7: LASSO Selected Predictors of log(OOP)",
    label="tab:rq7_lasso",
    filename="rq7_lasso_table.tex")

print(f"\nLASSO selected {len(selected)} / {len(feat_names)} features")
print(f"Best alpha: {lasso.alpha_:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(8, max(5, len(selected) * 0.3)))
top = selected.head(20)
colors = [RED if v > 0 else BLUE for v in top]
ax.barh(top.index, top.values, color=colors, edgecolor="black")
ax.set_xlabel("LASSO Coefficient")
ax.set_title("Top 20 LASSO-Selected Predictors")
ax.axvline(0, color="black", lw=1)
plt.tight_layout()
plt.savefig(f"{OUT}/rq7_lasso_predictors.png", dpi=150)
plt.close()
print("  → Figure saved: rq7_lasso_predictors.png")


# ══════════════════════════════════════════════════════════
# CELL 6 — RQ8: OP–IP Gap Analysis
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RQ8: OUTPATIENT vs INPATIENT GAP ANALYSIS")
print("=" * 70)

q8 = df[["net_oope_op", "net_oope_ip", "is_insured", "is_urban",
          "is_male", "age", "household_size", "wealth_10k",
          "district_uhc_index", "ip_private"]].dropna()
q8["gap"] = q8["net_oope_ip"] - q8["net_oope_op"]

formula_q8 = """
gap ~ is_insured + is_urban + is_male + age + household_size
+ wealth_10k + district_uhc_index + ip_private
"""
model_q8 = smf.ols(formula_q8, data=q8).fit(cov_type="HC3")
print(model_q8.summary())

gap_vars = ["is_insured", "is_urban", "is_male", "age",
            "household_size", "wealth_10k", "district_uhc_index", "ip_private"]
gap_results = pd.DataFrame({
    "Coefficient": model_q8.params[gap_vars].round(3),
    "Std Error":   model_q8.bse[gap_vars].round(3),
    "p-value":     model_q8.pvalues[gap_vars].round(4)
})
gap_results.to_csv(f"{OUT}/rq8_gap_results.csv")
results_to_latex(gap_results,
    caption="RQ8: OLS Regression — Determinants of IP--OP Expenditure Gap",
    label="tab:rq8_gap",
    filename="rq8_gap_table.tex")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(q8["gap"].clip(-10000, 50000), bins=60, color=BLUE, edgecolor="black", alpha=.7)
axes[0].axvline(0, color=RED, lw=2, ls="--")
axes[0].set_xlabel("IP − OP Expenditure (₹)")
axes[0].set_title("Distribution of IP–OP Gap")

sig = gap_results[gap_results["p-value"] < 0.05]
colors = [RED if v > 0 else GREEN for v in sig["Coefficient"]]
axes[1].barh(sig.index, sig["Coefficient"], color=colors, edgecolor="black")
axes[1].set_xlabel("Coefficient (₹)")
axes[1].set_title("Significant Gap Predictors (p < 0.05)")
plt.tight_layout()
plt.savefig(f"{OUT}/rq8_op_ip_gap.png", dpi=150)
plt.close()
print("  → Figure saved: rq8_op_ip_gap.png")


# ══════════════════════════════════════════════════════════
# CELL 7 — EXPORT SUMMARY DATASET
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPORTING ANALYSIS DATASET SUMMARY")
print("=" * 70)

summary_cols = [
    "state", "state_region", "district", "rural_urban",
    "age", "gender", "education", "occupation", "caste", "religion",
    "household_size", "has_insurance", "is_insured",
    "total_hh_spend", "total_oope", "net_oope_op", "net_oope_ip",
    "catastrophic", "district_uhc_index", "district_uhc_tercile",
    "op_private", "ip_private"
]
summary_cols = [c for c in summary_cols if c in df.columns]
df[summary_cols].to_csv(f"{OUT}/analysis_dataset_summary.csv", index=False)

import shutil
shutil.make_archive("oop_outputs", "zip", "oop_outputs")

print("All outputs saved in oop_outputs/")
print("Zip file created: oop_outputs.zip")
print("\n✓ Analysis complete!")
