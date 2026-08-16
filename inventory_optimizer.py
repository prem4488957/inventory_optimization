# inventory_optimizer.py
# Stages 10-13 - Reusable demand forecasting and inventory optimization functions.
# Consolidated, importable versions of the logic that previously lived inline in app.py.
import pandas as pd
import numpy as np
import pickle
import os
from scipy.stats import norm

# Z-score lookup for common service levels (cycle-service-level -> z)
Z_SCORES = {90: 1.28, 95: 1.65, 98: 2.05, 99: 2.33}

DEFAULT_FEATURES = [
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


# ---------------------------------------------------------------------------
# Stage 10 - Demand Forecasting
# ---------------------------------------------------------------------------
def load_engineered_data(path="featured_sales_data.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def z_for_service_level(service_level_pct):
    """Return the standard-normal z value for a target cycle-service-level %."""
    service_level_pct = int(round(service_level_pct))
    if service_level_pct in Z_SCORES:
        return Z_SCORES[service_level_pct]
    return float(norm.ppf(service_level_pct / 100.0))


def build_future_features(prod_df, horizon=14, unit_price=None, on_promo=0, features=DEFAULT_FEATURES):
    """
    Build a feature DataFrame for the next `horizon` days given a product's
    full historical DataFrame (must contain engineered columns and be sorted by date).
    Mirrors the recursive rolling-feature logic used by the ML forecasters.
    """
    prod_df = prod_df.sort_values("date").reset_index(drop=True)
    if unit_price is None:
        unit_price = float(prod_df["unit_price"].iloc[-1])

    last_date = pd.to_datetime(prod_df["date"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    sales = prod_df["units_sold"].values
    recent_sales = list(sales)
    recent_promo = list(prod_df["on_promo"].values)
    # Precompute last known rolling values per window from the historical series
    rolling = {}
    for w in [3, 7, 14, 30]:
        rolling[f"mean_{w}"] = prod_df[f"rolling_mean_{w}"].bfill().tolist()
        rolling[f"std_{w}"] = prod_df[f"rolling_std_{w}"].bfill().tolist()
    rolling["min_7"] = prod_df["rolling_min_7"].bfill().tolist()
    rolling["max_7"] = prod_df["rolling_max_7"].bfill().tolist()

    records = []
    for f_date in future_dates:
        dow, month = f_date.dayofweek, f_date.month
        records.append({
            "day_of_week": dow,
            "month": month,
            "quarter": f_date.quarter,
            "is_weekend": int(dow in [5, 6]),
            "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
            "month_sin": float(np.sin(2 * np.pi * month / 12)),
            "month_cos": float(np.cos(2 * np.pi * month / 12)),
            "on_promo": on_promo,
            "on_promo_lag7": recent_promo[-7],
            "unit_price": unit_price,
            "sales_lag_1": recent_sales[-1],
            "sales_lag_7": recent_sales[-7],
            "sales_lag_14": recent_sales[-14],
            "sales_lag_30": recent_sales[-30],
            "rolling_mean_3": rolling["mean_3"][-1],
            "rolling_std_3": rolling["std_3"][-1],
            "rolling_mean_7": rolling["mean_7"][-1],
            "rolling_std_7": rolling["std_7"][-1],
            "rolling_mean_14": rolling["mean_14"][-1],
            "rolling_std_14": rolling["std_14"][-1],
            "rolling_mean_30": rolling["mean_30"][-1],
            "rolling_std_30": rolling["std_30"][-1],
            "rolling_min_7": rolling["min_7"][-1],
            "rolling_max_7": rolling["max_7"][-1],
        })
        # advance the context window for the next day (values unknown -> carry forward last known)
        recent_sales.append(recent_sales[-1])
        recent_promo.append(recent_promo[-1])
        for w in [3, 7, 14, 30]:
            rolling[f"mean_{w}"].append(rolling[f"mean_{w}"][-1])
            rolling[f"std_{w}"].append(rolling[f"std_{w}"][-1])
        rolling["min_7"].append(rolling["min_7"][-1])
        rolling["max_7"].append(rolling["max_7"][-1])

    return pd.DataFrame(records), future_dates


def forecast_demand(product_id, model=None, horizon=14, features=DEFAULT_FEATURES, model_file=None, df=None):
    """
    Forecast daily demand for a product using an ML model (pickle file or object).
    Returns (future_features_df, future_dates, forecast_values).
    """
    if df is None:
        df = load_engineered_data()
    prod_df = df[df["product_id"] == product_id]

    if model is None:
        if model_file is None:
            raise ValueError("Provide either a `model` object or a `model_file` path.")
        with open(model_file, "rb") as f:
            model = pickle.load(f)

    future_features_df, future_dates = build_future_features(prod_df, horizon=horizon)
    preds = model.predict(future_features_df[features])
    preds = np.clip(preds, 0, None)
    return future_features_df, future_dates, np.asarray(preds)


# ---------------------------------------------------------------------------
# Stages 11-13 - Safety Stock, Reorder Point, EOQ
# ---------------------------------------------------------------------------
def safety_stock(forecast_std, lead_time_days, service_level_pct):
    """Safety stock = z * sigma_daily * sqrt(lead_time)."""
    z = z_for_service_level(service_level_pct)
    sigma = float(forecast_std) if float(forecast_std) > 0 else 2.0
    return z * sigma * np.sqrt(lead_time_days)


def lead_time_demand(avg_daily_demand, lead_time_days):
    """Expected demand during the supplier lead time."""
    return float(avg_daily_demand) * lead_time_days


def reorder_point(avg_daily_demand, lead_time_days, forecast_std, service_level_pct):
    """ROP = demand during lead time + safety stock."""
    ltd = lead_time_demand(avg_daily_demand, lead_time_days)
    ss = safety_stock(forecast_std, lead_time_days, service_level_pct)
    return ltd + ss


def economic_order_quantity(annual_demand, ordering_cost, holding_cost_per_unit_per_year):
    """EOQ = sqrt(2 * D * S / H)."""
    if annual_demand <= 0 or holding_cost_per_unit_per_year <= 0:
        return 0.0
    return float(np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit_per_year))


def days_of_supply(current_stock, avg_daily_demand):
    """How many days the current stock covers at the forecast demand rate."""
    return current_stock / (avg_daily_demand if avg_daily_demand > 0 else 1.0)


def restock_decision(current_stock, rop, avg_daily_demand, coverage_days=7):
    """Return (needs_restock, recommended_order_qty)."""
    needs_restock = current_stock < rop
    order_qty = 0
    if needs_restock:
        order_qty = int(np.ceil((rop + (avg_daily_demand * coverage_days)) - current_stock))
    return needs_restock, order_qty


def summarize_optimization(product_id, df=None, model=None, model_file=None, horizon=14,
                           current_stock=60, service_level_pct=95, ordering_cost=50.0,
                           holding_cost_percent=0.25, features=DEFAULT_FEATURES):
    """
    End-to-end convenience: forecast demand then compute the full inventory KPIs
    (safety stock, ROP, EOQ, days of supply, restock recommendation) for one product.
    """
    if df is None:
        df = load_engineered_data()
    prod_df = df[df["product_id"] == product_id]
    lead_time = int(prod_df["lead_time_days"].iloc[0])
    unit_price = float(prod_df["unit_price"].iloc[-1])
    name = prod_df["product_name"].iloc[0]
    category = prod_df["category"].iloc[0]

    _, future_dates, preds = forecast_demand(
        product_id, model=model, model_file=model_file, horizon=horizon, features=features, df=df
    )
    avg_daily = float(np.mean(preds))
    std_daily = float(np.std(preds))

    ss = safety_stock(std_daily, lead_time, service_level_pct)
    rop = reorder_point(avg_daily, lead_time, std_daily, service_level_pct)
    annual_demand = avg_daily * 365
    holding_cost = unit_price * holding_cost_percent
    eoq = economic_order_quantity(annual_demand, ordering_cost, holding_cost)
    dos = days_of_supply(current_stock, avg_daily)
    needs_restock, order_qty = restock_decision(current_stock, rop, avg_daily)

    return {
        "product_id": product_id,
        "product_name": name,
        "category": category,
        "lead_time_days": lead_time,
        "unit_price": unit_price,
        "horizon_days": horizon,
        "avg_daily_demand": round(avg_daily, 2),
        "forecast_std": round(std_daily, 2),
        "safety_stock": round(ss, 2),
        "reorder_point": round(rop, 2),
        "eoq": round(eoq, 2),
        "annual_demand": round(annual_demand, 2),
        "days_of_supply": round(dos, 2),
        "needs_restock": bool(needs_restock),
        "recommended_order_qty": order_qty,
        "forecast": [round(float(v), 2) for v in preds],
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
    }


if __name__ == "__main__":
    # Quick self-test using the best ML model
    data = load_engineered_data()
    summary = summarize_optimization("P101", df=data, model_file="random_forest_model.pkl")
    for k, v in summary.items():
        print(f"{k}: {v}")
