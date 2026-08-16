# accuracy_by_epoch.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def main():
    df = pd.read_csv("featured_sales_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    features = [
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
    target = "units_sold"
    split_date = "2025-10-01"
    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    # Small train/val split for monitoring accuracy during training
    val_df = train_df.tail(200)
    train_df = train_df.iloc[:-200]
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]

    results = {}

    # --- LightGBM with per-iteration evaluation ---
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    evals = {}
    model = lgb.train(
        {"objective": "regression", "metric": ["l1", "l2"], "learning_rate": 0.03, "num_leaves": 31, "seed": 42},
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[lgb.record_evaluation(evals), lgb.log_evaluation(0)],
    )
    print("EVAL KEYS:", list(evals["val"].keys()))
    metric_key = "rmse" if "rmse" in evals["val"] else "l2"
    lgb_mae = evals["val"]["l1"]
    lgb_rmse = np.sqrt(evals["val"]["l2"])
    # Final test
    pred = model.predict(X_test, num_iteration=model.best_iteration)
    results["LightGBM"] = {
        "epochs": list(range(1, len(lgb_mae) + 1)),
        "val_mae": lgb_mae,
        "val_rmse": lgb_rmse,
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_rmse": float(root_mean_squared_error(y_test, pred)),
        "best_epoch": int(np.argmin(lgb_mae)) + 1,
    }

    # --- GradientBoosting with staged prediction for per-epoch (iteration) metrics ---
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_epochs = list(range(1, gb.n_estimators + 1))
    gb_mae, gb_rmse = [], []
    for p in gb.staged_predict(X_val):
        gb_mae.append(mean_absolute_error(y_val, p))
        gb_rmse.append(root_mean_squared_error(y_val, p))
    pred = gb.predict(X_test)
    results["Gradient Boosting"] = {
        "epochs": gb_epochs,
        "val_mae": gb_mae,
        "val_rmse": gb_rmse,
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_rmse": float(root_mean_squared_error(y_test, pred)),
        "best_epoch": int(np.argmin(gb_mae)) + 1,
    }

    # --- Summary table of accuracy (% reduction in error) and precision ---
    print(f"{'Model':<18}{'Test MAE':>12}{'Test RMSE':>12}{'BestEpoch':>10}{'Prec@20%':>10}")
    lgb_best_epoch = results["LightGBM"]["best_epoch"]
    pred_lgb = model.predict(X_test, num_iteration=lgb_best_epoch)
    for name, r in results.items():
        pred_full = {"LightGBM": pred_lgb, "Gradient Boosting": gb.predict(X_test)}[name]
        tol = np.abs(y_test) * 0.20 + 1e-9
        prec = np.mean(np.abs(pred_full - y_test) <= tol) * 100
        print(f"{name:<18}{r['test_mae']:>12.3f}{r['test_rmse']:>12.3f}{r['best_epoch']:>10}{prec:>9.1f}%")

    print()
    for name, r in results.items():
        print(f"--- {name} per-epoch (validation) ---")
        print(f"{'Epoch':>6}{'Val MAE':>12}{'Val RMSE':>12}")
        n = len(r["epochs"])
        for i in range(0, n, max(1, n // 15)):
            e = r["epochs"][i]
            vmae = r["val_mae"][i]
            vrmse = r["val_rmse"][i]
            print(f"{e:>6}{vmae:>12.3f}{vrmse:>12.3f}")
        if r["epochs"][-1] not in [r["epochs"][i] for i in range(0, n, max(1, n // 15))]:
            print(f"{r['epochs'][-1]:>6}{r['val_mae'][-1]:>12.3f}{r['val_rmse'][-1]:>12.3f}")

    return results

if __name__ == "__main__":
    main()
