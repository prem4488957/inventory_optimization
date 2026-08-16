# seasonal_model.py
# Stage 6 - Traditional time-series seasonal baseline (statsmodels Holt-Winters / seasonal decomposition).
# Plays the role of the "Prophet" style forecasting baseline without requiring Prophet's cmdstan toolchain.
# A per-product multiplicative Holt-Winters exponential smoothing model with weekly seasonality.
import pandas as pd
import numpy as np
import pickle
import json
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

warnings.filterwarnings("ignore")


def train():
    print("[+] Loading engineered data...")
    df = pd.read_csv("featured_sales_data.csv")
    df["date"] = pd.to_datetime(df["date"])

    split_date = "2025-10-01"
    test_df = df[df["date"] >= split_date]

    results = {}
    fitted = {}

    for product_id, g in df.groupby("product_id"):
        g = g.sort_values("date").reset_index(drop=True)
        train_series = g.loc[g["date"] < split_date, "units_sold"].astype(float).reset_index(drop=True)
        test_series = g.loc[g["date"] >= split_date, "units_sold"].astype(float).reset_index(drop=True)

        # Multiplicative Holt-Winters with weekly (7-day) seasonality.
        # Seasonality period must be < len(series); 7 is safely inside ~600+ training points.
        try:
            model = ExponentialSmoothing(
                train_series,
                trend="add",
                seasonal="mul",
                seasonal_periods=7,
                initialization_method="estimated",
            ).fit(optimized=True)
            # Forecast length = number of test days for this product
            forecast = model.forecast(len(test_series))
        except Exception as e:
            # Fall back to a plain seasonal-naive baseline if HW fails to converge
            print(f"   [!] HW failed for {product_id}: {e} -- using seasonal-naive fallback.")
            weekly = np.tile(train_series.tail(7).to_numpy(), int(np.ceil(len(test_series) / 7)))[:len(test_series)]
            forecast = pd.Series(weekly)
            model = None

        forecast = np.clip(forecast.to_numpy(), 0, None)
        y_test = test_series.to_numpy()

        mae = mean_absolute_error(y_test, forecast)
        rmse = root_mean_squared_error(y_test, forecast)

        results[product_id] = {"MAE": round(float(mae), 3), "RMSE": round(float(rmse), 3), "name": g["product_name"].iloc[0]}
        fitted[product_id] = model
        print(f"   * {g['product_name'].iloc[0]}: MAE={mae:.2f} | RMSE={rmse:.2f}")

    # Save fitted models (per product) for later forecasting use
    with open("seasonal_models.pkl", "wb") as f:
        pickle.dump(fitted, f)
    print("[+] Saved per-product seasonal models to 'seasonal_models.pkl'")

    # Aggregate overall metrics (pool all test points)
    all_y = []
    all_p = []
    for product_id, g in df.groupby("product_id"):
        g = g.sort_values("date").reset_index(drop=True)
        test_series = g.loc[g["date"] >= split_date, "units_sold"].astype(float).reset_index(drop=True)
        mod = fitted[product_id]
        if mod is not None:
            fc = np.clip(mod.forecast(len(test_series)).to_numpy(), 0, None)
        else:
            weekly = np.tile(g.loc[g["date"] < split_date, "units_sold"].tail(7).to_numpy(), int(np.ceil(len(test_series) / 7)))[:len(test_series)]
            fc = np.clip(weekly, 0, None)
        all_y.extend(test_series.tolist())
        all_p.extend(fc.tolist())

    overall_mae = mean_absolute_error(all_y, all_p)
    overall_rmse = root_mean_squared_error(all_y, all_p)

    print(f"\n[+] Overall Seasonal Baseline: MAE={overall_mae:.3f} | RMSE={overall_rmse:.3f}")

    # Merge into model_metrics.json alongside the ML models
    with open("model_metrics.json", "r") as f:
        metrics_report = json.load(f)

    metrics_report["metrics"]["Seasonal Baseline (Holt-Winters)"] = {
        "MAE": round(float(overall_mae), 3),
        "RMSE": round(float(overall_rmse), 3),
        "per_product": results,
    }

    # Re-rank by MAE (ignore keys that aren't scalar MAE dicts)
    scalar = {k: v for k, v in metrics_report["metrics"].items()
              if isinstance(v, dict) and "MAE" in v and isinstance(v["MAE"], (int, float))}
    rankings = sorted(scalar.items(), key=lambda kv: kv[1]["MAE"])
    metrics_report["best_model"] = rankings[0][0]

    with open("model_metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    print("[+] Updated 'model_metrics.json' with seasonal baseline.")
    print(f"[+] Best model overall is now: {rankings[0][0]}\n")


if __name__ == "__main__":
    train()
