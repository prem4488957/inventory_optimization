# train_model.py
import pandas as pd
import numpy as np
import pickle
import json
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def train():
    print("[+] Loading engineered data...")
    df = pd.read_csv("featured_sales_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Feature selection
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
    
    # 2. Time-based Split: Train on 2024 to mid-2025, Test on late-2025
    split_date = "2025-10-01"
    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print(f"[+] Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")
    
    # 3. Define candidate models
    models = {
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            random_state=42
        ),
        "Linear Regression": LinearRegression(),
        "Support Vector (RBF)": SVR(C=1.0, epsilon=0.1),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n[+] Training {name}...")
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        predictions = np.clip(predictions, 0, None)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        
        results[name] = {"MAE": round(float(mae), 3), "RMSE": round(float(rmse), 3)}
        print(f"   • MAE: {mae:.2f} units | RMSE: {rmse:.2f} units")
        
        # 5. Save each trained model to disk
        slug = name.lower().replace(" ", "_").replace("_rbf_", "_").replace("(rbf)", "svr")
        filename = f"{slug}_model.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"   [+] Saved as '{filename}'")
    
    # 6. Rank models by MAE and save a metrics report
    rankings = sorted(results.items(), key=lambda kv: kv[1]["MAE"])
    with open("model_metrics.json", "w") as f:
        json.dump(
            {
                "split_date": split_date,
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "metrics": results,
                "best_model": rankings[0][0]
            },
            f,
            indent=2
        )
    
    print("\n[+] Model Ranking (by MAE):")
    for i, (name, m) in enumerate(rankings, 1):
        print(f"   {i}. {name}: MAE={m['MAE']} | RMSE={m['RMSE']}")
    
    print(f"\n[+] Best model: {rankings[0][0]}")
    print("[+] 'model_metrics.json' saved with all accuracies.\n")

if __name__ == "__main__":
    train()
