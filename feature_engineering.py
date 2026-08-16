# feature_engineering.py
import pandas as pd
import numpy as np

def build_features():
    print("[+] Loading raw sales data...")
    df = pd.read_csv("retail_sales_data.csv")
    
    # Convert date to datetime and sort sequentially
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)
    
    print("[+] Engineering calendar features...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclic seasonal encoding (helps tree models capture periodic seasonality)
    dow_sin = np.sin(2 * np.pi * df["day_of_week"] / 7)
    dow_cos = np.cos(2 * np.pi * df["day_of_week"] / 7)
    month_sin = np.sin(2 * np.pi * df["month"] / 12)
    month_cos = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"], df["dow_cos"] = dow_sin, dow_cos
    df["month_sin"], df["month_cos"] = month_sin, month_cos

    # Promo intensity: whether the same day last week was on promo (demand carryover)
    df["on_promo_lag7"] = df.groupby("product_id")["on_promo"].shift(7).fillna(0)

    print("[+] Engineering lag and rolling statistics...")
    # Lag Features
    df["sales_lag_1"] = df.groupby("product_id")["units_sold"].shift(1)
    df["sales_lag_7"] = df.groupby("product_id")["units_sold"].shift(7)
    df["sales_lag_14"] = df.groupby("product_id")["units_sold"].shift(14)
    df["sales_lag_30"] = df.groupby("product_id")["units_sold"].shift(30)

    # Rolling Window Features (shifted by 1 to avoid leakage)
    for w in [3, 7, 14, 30]:
        df[f"rolling_mean_{w}"] = df.groupby("product_id")["units_sold"].transform(
            lambda x: x.shift(1).rolling(w).mean())
        df[f"rolling_std_{w}"] = df.groupby("product_id")["units_sold"].transform(
            lambda x: x.shift(1).rolling(w).std())
        df[f"rolling_min_{w}"] = df.groupby("product_id")["units_sold"].transform(
            lambda x: x.shift(1).rolling(w).min())
        df[f"rolling_max_{w}"] = df.groupby("product_id")["units_sold"].transform(
            lambda x: x.shift(1).rolling(w).max())

    # Drop rows containing NaN values created by initial lag shifts
    initial_rows = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    print(f"[+] Cleaned initial lag NaNs: {initial_rows} -> {len(df_clean)} rows remaining.")
    
    # Save the processed dataset
    df_clean.to_csv("featured_sales_data.csv", index=False)
    print("[+] Success! 'featured_sales_data.csv' is ready.\n")

if __name__ == "__main__":
    build_features()