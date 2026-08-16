# explainability.py
# Stage 20 - Explainable AI (XAI) using SHAP.
# Provides global feature importance (bar), a SHAP summary (beeswarm) plot,
# and a local explanation (waterfall) for an individual prediction.
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = [
    "day_of_week", "month", "quarter", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "on_promo", "on_promo_lag7", "unit_price",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
    "rolling_mean_3", "rolling_std_3",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_14", "rolling_std_14",
    "rolling_mean_30", "rolling_std_30",
    "rolling_min_7", "rolling_max_7",
]

FEATURE_LABELS = {
    "day_of_week": "Day of Week",
    "month": "Month",
    "quarter": "Quarter",
    "is_weekend": "Weekend",
    "dow_sin": "Weekday (sin)",
    "dow_cos": "Weekday (cos)",
    "month_sin": "Month (sin)",
    "month_cos": "Month (cos)",
    "on_promo": "On Promo",
    "on_promo_lag7": "Promo last week",
    "unit_price": "Unit Price",
    "sales_lag_1": "Sales Yesterday",
    "sales_lag_7": "Sales 7d ago",
    "sales_lag_14": "Sales 14d ago",
    "sales_lag_30": "Sales 30d ago",
    "rolling_mean_3": "3d Avg",
    "rolling_std_3": "3d Std",
    "rolling_mean_7": "7d Avg",
    "rolling_std_7": "7d Std",
    "rolling_mean_14": "14d Avg",
    "rolling_std_14": "14d Std",
    "rolling_mean_30": "30d Avg",
    "rolling_std_30": "30d Std",
    "rolling_min_7": "7d Min",
    "rolling_max_7": "7d Max",
}


def load_model(model_file):
    with open(model_file, "rb") as f:
        return pickle.load(f)


def prepare_sample(df, n_samples=500, seed=42):
    """Return a sample of the engineered data for SHAP background/summary."""
    np.random.seed(seed)
    sample = df[FEATURES].sample(n=min(n_samples, len(df)), random_state=seed)
    return sample


def shap_values(model, X, n_samples=200):
    """Compute SHAP values using the most efficient explainer for the model."""
    X = X.head(n_samples) if isinstance(X, pd.DataFrame) else X[:n_samples]
    explainer = shap.TreeExplainer(model)
    return explainer, explainer.shap_values(X)


def global_importance_bar(model, X, save_path="shap_importance.png", top_n=12):
    """Mean |SHAP| bar chart -> top features driving predictions."""
    explainer, sv = shap_values(model, X)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    labels = [FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in order[:top_n]]
    vals = mean_abs[order[:top_n]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels[::-1], vals[::-1], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value| (impact on forecasted units)")
    ax.set_title("Global Feature Importance (SHAP)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return save_path


def summary_beeswarm(model, X, save_path="shap_summary.png", top_n=12):
    """SHAP beeswarm summary plot."""
    explainer, sv = shap_values(model, X)
    feature_names = [FEATURE_LABELS.get(f, f) for f in FEATURES]
    X_named = X.copy()
    X_named.columns = feature_names
    fig = shap.summary_plot(sv, X_named, max_display=top_n, show=False)
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()
    return save_path


def local_waterfall(model, X, row_index=0, save_path="shap_local.png"):
    """Waterfall explanation for one specific forecast."""
    explainer, _ = shap_values(model, X, n_samples=len(X))
    sv = explainer.shap_values(X)
    feature_names = [FEATURE_LABELS.get(f, f) for f in FEATURES]
    X_named = X.copy()
    X_named.columns = feature_names
    exp = shap.Explanation(values=sv[row_index], base_values=explainer.expected_value,
                            data=X_named.iloc[row_index].values, feature_names=feature_names)
    fig = shap.waterfall_plot(exp, show=False)
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()
    return save_path


def feature_importance_df(model, X):
    """Return a sorted DataFrame of mean |SHAP| per feature."""
    explainer, sv = shap_values(model, X)
    mean_abs = np.abs(sv).mean(axis=0)
    df_out = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURES],
        "Mean |SHAP|": np.round(mean_abs, 4),
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    return df_out


def generate_explanation(product_name, explainer, local_sv, local_feats,
                         imp_df, top_n=3):
    """Build a plain-English paragraph explaining the SHAP graphs for a product.

    local_sv : SHAP values for the next-day forecast row (1D array).
    local_feats : renamed next-day feature row (for context values).
    imp_df : global mean-|SHAP| importance DataFrame.
    """
    base = float(explainer.expected_value)
    pred = base + float(local_sv.sum())
    local_sv = np.asarray(local_sv).reshape(-1)

    pos = [(i, float(v)) for i, v in enumerate(local_sv) if v > 0]
    neg = [(i, float(v)) for i, v in enumerate(local_sv) if v < 0]
    pos.sort(key=lambda x: x[1], reverse=True)
    neg.sort(key=lambda x: x[1])

    names = list(local_feats.columns)

    def describe(idx, val):
        label = names[idx]
        feat_val = local_feats.iloc[0][label]
        if isinstance(feat_val, float):
            feat_val = round(feat_val, 2)
        direction = "pushes" if val > 0 else "pulls"
        return f"**{label}** ({feat_val}) {direction} the forecast {abs(val):.1f} units"

    lines = [
        f"For **{product_name}**, the model's next-day baseline demand is "
        f"**{base:.1f} units**. After accounting for the product's current conditions, "
        f"the forecast lands at **{pred:.1f} units**.",
    ]

    if pos:
        top_pos = ", ".join(describe(i, v) for i, v in pos[:top_n])
        lines.append(f"The strongest upward drivers are {top_pos}.")
    if neg:
        top_neg = ", ".join(describe(i, v) for i, v in neg[:top_n])
        lines.append(f"The strongest downward pressures are {top_neg}.")

    global_top = imp_df["Feature"].head(top_n).tolist()
    lines.append(
        f"Across the whole catalog, the features that generally matter most to this model are "
        f"{', '.join(f'**{g}**' for g in global_top)} (see the summary plot on the right)."
    )
    return " ".join(lines)


if __name__ == "__main__":
    from inventory_optimizer import load_engineered_data
    data = load_engineered_data()
    model = load_model("xgboost_model.pkl")
    X = prepare_sample(data)
    print(f"Explaining {type(model).__name__} over {len(X)} samples\n")
    print(feature_importance_df(model, X).to_string(index=False))
    global_importance_bar(model, X)
    print("\nSaved: shap_importance.png, shap_summary.png")