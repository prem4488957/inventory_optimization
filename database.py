# database.py
# Stage 15 - SQLite database persistence.
# Stores products, daily sales, forecasts, and inventory optimization results.
import sqlite3
import pandas as pd
from contextlib import closing

DB_PATH = "inventory.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create tables if they don't already exist."""
    with closing(get_connection()) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                product_id   TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,
                category     TEXT NOT NULL,
                lead_time_days INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sales (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL,
                product_id  TEXT NOT NULL,
                on_promo    INTEGER NOT NULL,
                unit_price  REAL NOT NULL,
                units_sold  INTEGER NOT NULL,
                UNIQUE(date, product_id)
            );

            CREATE TABLE IF NOT EXISTS forecasts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id  TEXT NOT NULL,
                model_name  TEXT NOT NULL,
                forecast_date TEXT NOT NULL,
                predicted_units REAL NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS inventory_optimization (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id    TEXT NOT NULL,
                model_name    TEXT NOT NULL,
                avg_daily_demand REAL NOT NULL,
                safety_stock  REAL NOT NULL,
                reorder_point REAL NOT NULL,
                eoq           REAL NOT NULL,
                days_of_supply REAL NOT NULL,
                needs_restock INTEGER NOT NULL,
                recommended_order_qty INTEGER NOT NULL,
                created_at    TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()


def seed_products_from_df(df):
    """Insert unique product rows from the engineered DataFrame."""
    with closing(get_connection()) as conn:
        for _, row in df[["product_id", "product_name", "category", "lead_time_days"]].drop_duplicates().iterrows():
            conn.execute(
                "INSERT OR IGNORE INTO products (product_id, product_name, category, lead_time_days) "
                "VALUES (?, ?, ?, ?)",
                (row["product_id"], row["product_name"], row["category"], int(row["lead_time_days"])),
            )
        conn.commit()


def seed_sales_from_df(df):
    """Insert daily sales rows from the engineered DataFrame (deduplicated)."""
    with closing(get_connection()) as conn:
        rows = df[["date", "product_id", "on_promo", "unit_price", "units_sold"]].to_records(index=False)
        conn.executemany(
            "INSERT OR IGNORE INTO sales (date, product_id, on_promo, unit_price, units_sold) "
            "VALUES (?, ?, ?, ?, ?)",
            [(str(r.date.date() if hasattr(r.date, "date") else r.date), r.product_id,
              int(r.on_promo), float(r.unit_price), int(r.units_sold)) for r in rows],
        )
        conn.commit()


def save_forecast(product_id, model_name, forecast_dates, forecast_values):
    with closing(get_connection()) as conn:
        conn.executemany(
            "INSERT INTO forecasts (product_id, model_name, forecast_date, predicted_units) "
            "VALUES (?, ?, ?, ?)",
            [(product_id, model_name, str(d), float(v)) for d, v in zip(forecast_dates, forecast_values)],
        )
        conn.commit()


def save_optimization(opt: dict, model_name="Random Forest"):
    with closing(get_connection()) as conn:
        conn.execute(
            "INSERT INTO inventory_optimization "
            "(product_id, model_name, avg_daily_demand, safety_stock, reorder_point, eoq, "
            " days_of_supply, needs_restock, recommended_order_qty) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                opt["product_id"], model_name, opt["avg_daily_demand"], opt["safety_stock"],
                opt["reorder_point"], opt["eoq"], opt["days_of_supply"],
                int(opt["needs_restock"]), int(opt["recommended_order_qty"]),
            ),
        )
        conn.commit()


def read_table(table):
    with closing(get_connection()) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def setup(df):
    """Convenience: init DB and seed products + sales from the engineered DataFrame."""
    init_db()
    seed_products_from_df(df)
    seed_sales_from_df(df)


if __name__ == "__main__":
    from inventory_optimizer import load_engineered_data, summarize_optimization

    data = load_engineered_data()
    setup(data)
    print("DB initialized and seeded from engineered data.")

    # Persist one optimization run per product as a demo
    for pid in data["product_id"].unique():
        opt = summarize_optimization(pid, df=data, model_file="random_forest_model.pkl")
        save_optimization(opt)
        save_forecast(pid, "Random Forest", opt["forecast_dates"], opt["forecast"])
        print(f"Saved optimization for {pid}: ROP={opt['reorder_point']}, EOQ={opt['eoq']}")

    print("\n--- products ---")
    print(read_table("products").to_string(index=False))
    print("\n--- inventory_optimization (latest) ---")
    print(read_table("inventory_optimization").tail(5).to_string(index=False))
