# tests/test_multi_product.py
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inventory_optimizer import (
    load_engineered_data,
    forecast_demand,
    summarize_optimization,
    run_multi_product_optimization,
    classify_inventory_risk,
    get_deterministic_stock_for_product,
)


@pytest.fixture(scope="module")
def data():
    return load_engineered_data()


def test_classify_inventory_risk():
    # Stock < ROP -> HIGH
    risk, action, badge = classify_inventory_risk(10, 50, 2.0, 5)
    assert risk == "HIGH"
    assert action == "ORDER NOW"
    assert "HIGH" in badge

    # Stock between ROP and 1.3*ROP -> MEDIUM
    risk_m, action_m, _ = classify_inventory_risk(55, 50, 6.0, 3)
    assert risk_m == "MEDIUM"
    assert action_m == "REORDER SOON"

    # Stock >= 1.3*ROP and DOS >= 1.5*lead_time -> LOW
    risk_l, action_l, _ = classify_inventory_risk(100, 50, 10.0, 3)
    assert risk_l == "LOW"
    assert action_l == "STOCK OK"


def test_deterministic_stock():
    # Same product_id should return identical stock value
    s1 = get_deterministic_stock_for_product("P101", 45.0)
    s2 = get_deterministic_stock_for_product("P101", 45.0)
    assert s1 == s2
    assert s1 > 0


def test_run_multi_product_optimization_brand_filter(data):
    model_file = "random_forest_model.pkl"
    if not os.path.exists(model_file):
        pytest.skip("random_forest_model.pkl not present")

    res = run_multi_product_optimization(
        data,
        model_file=model_file,
        horizon=30,
        brand_filter="Apple",
        category_filter="All Categories",
    )

    res_df = res["results_df"]
    summary = res["summary"]

    assert not res_df.empty
    assert set(res_df["Brand"].unique()) == {"Apple"}
    assert summary["total_products"] == len(res_df)
    assert summary["high_risk_count"] + summary["medium_risk_count"] + summary["healthy_count"] == summary["total_products"]


def test_multi_product_single_product_consistency(data):
    """
    Requirement 21: Data Consistency Test
    Verify that single-product and multi-product calculation outputs match identically
    for the exact same product and parameters.
    """
    model_file = "random_forest_model.pkl"
    if not os.path.exists(model_file):
        pytest.skip("random_forest_model.pkl not present")

    product_id = "P101"
    horizon = 30
    service_level = 95
    stock_val = 60

    # 1. Single Product calculation
    single_res = summarize_optimization(
        product_id=product_id,
        df=data,
        model_file=model_file,
        horizon=horizon,
        current_stock=stock_val,
        service_level_pct=service_level,
    )

    # 2. Multi Product calculation with stock override for P101
    multi_out = run_multi_product_optimization(
        df=data,
        model_file=model_file,
        horizon=horizon,
        brand_filter="Apple",
        service_level_pct=service_level,
        stock_overrides={product_id: stock_val},
    )

    multi_product_detail = multi_out["product_details"][product_id]

    # Verify exact metrics match between single and multi product calculations
    assert single_res["avg_daily_demand"] == multi_product_detail["avg_daily_demand"]
    assert single_res["safety_stock"] == multi_product_detail["safety_stock"]
    assert single_res["reorder_point"] == multi_product_detail["reorder_point"]
    assert single_res["eoq"] == multi_product_detail["eoq"]
    assert single_res["days_of_supply"] == multi_product_detail["days_of_supply"]
    assert single_res["needs_restock"] == multi_product_detail["needs_restock"]
    assert single_res["recommended_order_qty"] == multi_product_detail["recommended_order_qty"]
    assert single_res["forecast"] == multi_product_detail["forecast"]
