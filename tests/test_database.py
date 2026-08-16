# tests/test_database.py
import os
import sqlite3
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from database import (
    init_db,
    get_connection,
    seed_products_from_df,
    seed_sales_from_df,
    save_forecast,
    save_optimization,
    read_table,
)

DB_PATH = "inventory.db"


def test_init_db_creates_tables():
    init_db()
    with get_connection() as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
    for t in ["products", "sales", "forecasts", "inventory_optimization"]:
        assert t in tables


def test_seed_and_read_products():
    init_db()
    df = pd.read_csv("featured_sales_data.csv")
    seed_products_from_df(df)
    prods = read_table("products")
    assert prods["product_id"].nunique() >= 5


def test_seed_sales_and_read():
    init_db()
    df = pd.read_csv("featured_sales_data.csv")
    seed_sales_from_df(df)
    sales = read_table("sales")
    assert len(sales) >= 3500  # 5 products * ~700 days


def test_save_and_read_forecast():
    init_db()
    # clear any rows from a previous run for idempotency
    with get_connection() as conn:
        conn.execute("DELETE FROM forecasts WHERE model_name='TestModel'")
        conn.commit()
    save_forecast("P101", "TestModel", ["2026-01-01", "2026-01-02"], [30.5, 29.8])
    fc = read_table("forecasts")
    latest = fc[fc["model_name"] == "TestModel"]
    assert len(latest) == 2
    assert latest["predicted_units"].sum() == pytest_approx(60.3)


def test_save_optimization_roundtrip():
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM inventory_optimization WHERE model_name='TestModel'")
        conn.commit()
    opt = {
        "product_id": "P101", "avg_daily_demand": 30.4, "safety_stock": 2.3,
        "reorder_point": 124.0, "eoq": 66.7, "days_of_supply": 2.0,
        "needs_restock": True, "recommended_order_qty": 278,
    }
    save_optimization(opt, model_name="TestModel")
    rows = read_table("inventory_optimization")
    rows = rows[rows["model_name"] == "TestModel"]
    assert len(rows) == 1
    assert rows.iloc[-1]["needs_restock"] == 1
    assert rows.iloc[-1]["reorder_point"] == pytest_approx(124.0)


def pytest_approx(v):
    from pytest import approx
    return approx(v)
