# inventory_optimizer.py
# Stages 10-13 - Reusable demand forecasting and inventory optimization functions.
# Consolidated, importable versions of the logic that previously lived inline in app.py.
import pandas as pd
import numpy as np
import pickle
import os
import hashlib
from scipy.stats import norm

try:
    from deep_learning_model import LSTMRegressor
except Exception:
    pass

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
    if not os.path.exists(path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, path)
    if not os.path.exists(path):
        try:
            from generate_data import generate_retail_data
            from feature_engineering import engineer_features
            raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retail_sales_data.csv")
            generate_retail_data(raw_path)
            engineer_features(raw_path, path)
        except Exception:
            pass
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
    """
    prod_df = prod_df.sort_values("date").reset_index(drop=True)
    if unit_price is None:
        unit_price = float(prod_df["unit_price"].iloc[-1])

    last_date = pd.to_datetime(prod_df["date"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    sales = list(prod_df["units_sold"].values)
    recent_promo = list(prod_df["on_promo"].values)

    records = []
    for f_date in future_dates:
        dow, month = f_date.dayofweek, f_date.month
        s_arr = np.array(sales)
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
            "on_promo_lag7": recent_promo[-7] if len(recent_promo) >= 7 else 0,
            "unit_price": unit_price,
            "sales_lag_1": float(s_arr[-1]),
            "sales_lag_7": float(s_arr[-7] if len(s_arr) >= 7 else s_arr[-1]),
            "sales_lag_14": float(s_arr[-14] if len(s_arr) >= 14 else s_arr[-1]),
            "sales_lag_30": float(s_arr[-30] if len(s_arr) >= 30 else s_arr[-1]),
            "rolling_mean_3": float(s_arr[-3:].mean()),
            "rolling_std_3": float(s_arr[-3:].std(ddof=0)),
            "rolling_mean_7": float(s_arr[-7:].mean()),
            "rolling_std_7": float(s_arr[-7:].std(ddof=0)),
            "rolling_mean_14": float(s_arr[-14:].mean()),
            "rolling_std_14": float(s_arr[-14:].std(ddof=0)),
            "rolling_mean_30": float(s_arr[-30:].mean()),
            "rolling_std_30": float(s_arr[-30:].std(ddof=0)),
            "rolling_min_7": float(s_arr[-7:].min()),
            "rolling_max_7": float(s_arr[-7:].max()),
        })
        sales.append(sales[-1])
        recent_promo.append(on_promo)

    return pd.DataFrame(records)[features], future_dates


def forecast_demand(product_id, model=None, horizon=14, features=DEFAULT_FEATURES, model_file=None, df=None):
    """
    Forecast daily demand for a product using an ML, Seasonal (Holt-Winters), or LSTM model.
    Returns (future_features_df, future_dates, forecast_values).
    """
    if df is None:
        df = load_engineered_data()
    prod_df = df[df["product_id"] == product_id].sort_values("date").reset_index(drop=True)

    if model is None:
        if model_file is None:
            raise ValueError("Provide either a `model` object or a `model_file` path.")
        with open(model_file, "rb") as f:
            model = pickle.load(f)

    last_date = pd.to_datetime(prod_df["date"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    unit_price = float(prod_df["unit_price"].iloc[-1])

    # Check for dict-based per-product models (Seasonal Holt-Winters or PyTorch LSTM)
    if isinstance(model, dict) and product_id in model:
        item = model[product_id]
        # PyTorch LSTM dict structure
        if isinstance(item, dict) and "model" in item and "scaler" in item:
            import torch
            l_mod = item["model"]
            l_scaler = item["scaler"]
            window = item.get("window", 30)
            train_s = prod_df["units_sold"].values
            scaled_s = l_scaler.transform(train_s.reshape(-1, 1)).ravel()
            ctx = list(scaled_s[-window:])
            preds = []
            l_mod.eval()
            with torch.no_grad():
                for _ in range(horizon):
                    x_in = torch.tensor(np.array(ctx[-window:], dtype=np.float32)).reshape(1, window, 1)
                    p_sc = l_mod(x_in).item()
                    p_orig = l_scaler.inverse_transform([[p_sc]])[0, 0]
                    p_val = max(0.0, float(p_orig))
                    preds.append(p_val)
                    ctx.append(p_sc)
            ff_df, _ = build_future_features(prod_df, horizon=horizon, features=features)
            return ff_df, future_dates, np.asarray(preds)
        
        # Seasonal Holt-Winters model object
        elif hasattr(item, "forecast"):
            raw_fc = item.forecast(horizon)
            preds = np.clip(np.asarray(raw_fc), 0, None)
            ff_df, _ = build_future_features(prod_df, horizon=horizon, features=features)
            return ff_df, future_dates, preds

    # Autoregressive multi-step recursive forecasting for tabular ML models
    sales = list(prod_df["units_sold"].values)
    recent_promo = list(prod_df["on_promo"].values)
    records = []
    preds = []

    for f_date in future_dates:
        dow, month = f_date.dayofweek, f_date.month
        s_arr = np.array(sales)
        row = {
            "day_of_week": dow,
            "month": month,
            "quarter": f_date.quarter,
            "is_weekend": int(dow in [5, 6]),
            "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
            "month_sin": float(np.sin(2 * np.pi * month / 12)),
            "month_cos": float(np.cos(2 * np.pi * month / 12)),
            "on_promo": 0,
            "on_promo_lag7": recent_promo[-7] if len(recent_promo) >= 7 else 0,
            "unit_price": unit_price,
            "sales_lag_1": float(s_arr[-1]),
            "sales_lag_7": float(s_arr[-7] if len(s_arr) >= 7 else s_arr[-1]),
            "sales_lag_14": float(s_arr[-14] if len(s_arr) >= 14 else s_arr[-1]),
            "sales_lag_30": float(s_arr[-30] if len(s_arr) >= 30 else s_arr[-1]),
            "rolling_mean_3": float(s_arr[-3:].mean()),
            "rolling_std_3": float(s_arr[-3:].std(ddof=0)),
            "rolling_mean_7": float(s_arr[-7:].mean()),
            "rolling_std_7": float(s_arr[-7:].std(ddof=0)),
            "rolling_mean_14": float(s_arr[-14:].mean()),
            "rolling_std_14": float(s_arr[-14:].std(ddof=0)),
            "rolling_mean_30": float(s_arr[-30:].mean()),
            "rolling_std_30": float(s_arr[-30:].std(ddof=0)),
            "rolling_min_7": float(s_arr[-7:].min()),
            "rolling_max_7": float(s_arr[-7:].max()),
        }
        records.append(row)
        single_row_df = pd.DataFrame([row])[features]
        if hasattr(model, "predict"):
            p_val = float(model.predict(single_row_df)[0])
        else:
            p_val = float(s_arr[-1])
        p_val = max(0.0, p_val)
        preds.append(p_val)
        sales.append(p_val)
        recent_promo.append(0)

    future_features_df = pd.DataFrame(records)[features]
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


def classify_inventory_risk(current_stock, rop, dos, lead_time_days):
    """
    Classify inventory risk based on transparent rules against ROP and Days of Supply.
    Returns (risk_level, action_str, risk_badge_string).
    """
    if current_stock < rop or dos < lead_time_days:
        risk_level = "HIGH"
        action = "ORDER NOW"
        badge = "🔴 HIGH RISK"
    elif current_stock < rop * 1.30 or dos < lead_time_days * 1.5:
        risk_level = "MEDIUM"
        action = "REORDER SOON"
        badge = "🟡 MEDIUM RISK"
    else:
        risk_level = "LOW"
        action = "STOCK OK"
        badge = "🟢 HEALTHY"

    return risk_level, action, badge


def get_deterministic_stock_for_product(product_id, rop):
    """
    Generate a realistic, deterministic initial stock level for a product
    based on its product_id hash and ROP, ensuring realistic High, Medium, and Healthy spread.
    """
    seed_val = int(hashlib.md5(product_id.encode('utf-8')).hexdigest(), 16)
    factor = (seed_val % 100) / 100.0  # float between 0.00 and 0.99

    if factor < 0.25:
        # High Risk (~25% of products)
        stock = int(np.round(rop * (0.35 + 0.58 * (factor / 0.25))))
    elif factor < 0.60:
        # Medium Risk (~35% of products)
        stock = int(np.round(rop * (1.02 + 0.25 * ((factor - 0.25) / 0.35))))
    else:
        # Healthy / Low Risk (~40% of products)
        stock = int(np.round(rop * (1.35 + 1.15 * ((factor - 0.60) / 0.40))))

    return max(5, stock)


def run_multi_product_optimization(
    df,
    model=None,
    model_file=None,
    horizon=30,
    brand_filter="All Brands",
    category_filter="All Categories",
    service_level_pct=95,
    ordering_cost=50.0,
    holding_cost_percent=0.25,
    stock_overrides=None,
    progress_callback=None
):
    """
    Run demand forecasting and inventory optimization across multiple products matching filters.
    Returns a dict containing:
      - 'results_df': pandas DataFrame with all product-level optimization metrics
      - 'summary': dict with aggregate portfolio metrics
      - 'product_details': dict mapping product_id to full summary dict
    """
    sub_df = df.copy()
    if brand_filter and brand_filter != "All Brands":
        sub_df = sub_df[sub_df["brand"] == brand_filter]
    if category_filter and category_filter != "All Categories":
        sub_df = sub_df[sub_df["category"] == category_filter]

    # Get unique products sorted by product_id
    product_records = sub_df[["product_id", "product_name", "brand", "category"]].drop_duplicates().to_dict("records")
    product_records = sorted(product_records, key=lambda x: x["product_id"])

    stock_overrides = stock_overrides or {}
    results = []
    product_details = {}
    total_prods = len(product_records)

    for idx, prec in enumerate(product_records):
        pid = prec["product_id"]
        if progress_callback:
            progress_callback(idx + 1, total_prods, prec["product_name"])

        try:
            prod_df = df[df["product_id"] == pid].sort_values("date").reset_index(drop=True)
            lead_time = int(prod_df["lead_time_days"].iloc[0])
            unit_price = float(prod_df["unit_price"].iloc[-1])
            if unit_price < 2000:
                unit_price *= 83.0  # Fail-safe USD -> INR conversion

            ff_df, future_dates, preds = forecast_demand(
                pid, model=model, model_file=model_file, horizon=horizon, df=df
            )
            avg_daily = float(np.mean(preds))
            std_daily = float(np.std(preds)) if float(np.std(preds)) > 0 else 2.0

            ss = safety_stock(std_daily, lead_time, service_level_pct)
            rop = reorder_point(avg_daily, lead_time, std_daily, service_level_pct)
            annual_demand = avg_daily * 365.0
            holding_cost = unit_price * holding_cost_percent
            eoq = economic_order_quantity(annual_demand, ordering_cost, holding_cost)

            # Determine stock
            if pid in stock_overrides:
                cur_stock = int(stock_overrides[pid])
            else:
                cur_stock = get_deterministic_stock_for_product(pid, rop)

            dos = days_of_supply(cur_stock, avg_daily)
            needs_restock, order_qty = restock_decision(cur_stock, rop, avg_daily)

            risk_level, action, risk_badge = classify_inventory_risk(cur_stock, rop, dos, lead_time)

            forecast_total = int(round(preds.sum()))

            res_row = {
                "product_id": pid,
                "Product": prec["product_name"],
                "Brand": prec["brand"],
                "Category": prec["category"],
                "Current Stock": cur_stock,
                "Avg Forecast/Day": round(avg_daily, 2),
                "Forecast Total": forecast_total,
                "Safety Stock": round(ss, 2),
                "Reorder Point": round(rop, 2),
                "Days of Supply": round(dos, 2),
                "Stock Risk": risk_badge,
                "Recommended Order": order_qty,
                "Action": action,
                "risk_level": risk_level,
                "eoq": round(eoq, 2),
                "lead_time_days": lead_time,
                "unit_price": round(unit_price, 2),
                "needs_restock": needs_restock,
                "status": "SUCCESS"
            }
            results.append(res_row)

            product_details[pid] = {
                "product_id": pid,
                "product_name": prec["product_name"],
                "brand": prec["brand"],
                "category": prec["category"],
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
                "risk_level": risk_level,
                "action": action,
                "risk_badge": risk_badge,
                "current_stock": cur_stock
            }

        except Exception as err:
            # Handle product failure gracefully per requirement 17
            results.append({
                "product_id": pid,
                "Product": prec["product_name"],
                "Brand": prec["brand"],
                "Category": prec["category"],
                "Current Stock": 0,
                "Avg Forecast/Day": 0.0,
                "Forecast Total": 0,
                "Safety Stock": 0.0,
                "Reorder Point": 0.0,
                "Days of Supply": 0.0,
                "Stock Risk": "❌ ERROR",
                "Recommended Order": 0,
                "Action": "Forecast Unavailable",
                "risk_level": "ERROR",
                "eoq": 0.0,
                "lead_time_days": 0,
                "unit_price": 0.0,
                "needs_restock": False,
                "status": f"FAILED: {err}"
            })

    res_df = pd.DataFrame(results)

    # Compute high-level portfolio summary metrics
    if not res_df.empty:
        high_risk_count = int((res_df["risk_level"] == "HIGH").sum())
        med_risk_count = int((res_df["risk_level"] == "MEDIUM").sum())
        low_risk_count = int((res_df["risk_level"] == "LOW").sum())
        total_rec_order = int(res_df["Recommended Order"].sum())
        total_fc_demand = int(res_df["Forecast Total"].sum())
    else:
        high_risk_count = med_risk_count = low_risk_count = total_rec_order = total_fc_demand = 0

    summary_kpis = {
        "total_products": total_prods,
        "high_risk_count": high_risk_count,
        "medium_risk_count": med_risk_count,
        "healthy_count": low_risk_count,
        "total_recommended_order": total_rec_order,
        "total_forecasted_demand": total_fc_demand,
    }

    return {
        "results_df": res_df,
        "summary": summary_kpis,
        "product_details": product_details
    }


if __name__ == "__main__":
    # Quick self-test using the best ML model
    data = load_engineered_data()
    summary = summarize_optimization("P101", df=data, model_file="random_forest_model.pkl")
    for k, v in summary.items():
        print(f"{k}: {v}")

