# tests/test_inventory_optimizer.py
import os
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inventory_optimizer import (
    z_for_service_level,
    safety_stock,
    lead_time_demand,
    reorder_point,
    economic_order_quantity,
    days_of_supply,
    restock_decision,
    load_engineered_data,
    forecast_demand,
    summarize_optimization,
)


@pytest.fixture(scope="module")
def data():
    return load_engineered_data()


def test_z_scores_known_values():
    assert z_for_service_level(90) == pytest.approx(1.28, abs=0.01)
    assert z_for_service_level(95) == pytest.approx(1.65, abs=0.01)
    assert z_for_service_level(99) == pytest.approx(2.33, abs=0.01)


def test_z_for_arbitrary_service_level():
    # interpolated value should be between 90 and 95 z's
    z = z_for_service_level(93)
    assert 1.28 < z < 1.65


def test_safety_stock_positive():
    ss = safety_stock(5.0, 4, 95)
    assert ss > 0
    # ss = 1.65 * 5 * sqrt(4) = 16.5
    assert ss == pytest.approx(16.5, abs=0.01)


def test_lead_time_demand():
    assert lead_time_demand(10.0, 5) == 50.0


def test_reorder_point_combines():
    rop = reorder_point(10.0, 5, 4.0, 95)
    # ltd = 50, ss = 1.65*4*sqrt(5) = 14.76 -> rop ~ 64.76
    assert rop > 50
    assert rop == pytest.approx(50 + 1.65 * 4 * np.sqrt(5), abs=0.01)


def test_eoq_formula():
    # EOQ = sqrt(2*D*S/H)
    eoq = economic_order_quantity(1000, 50, 10)
    assert eoq == pytest.approx(np.sqrt(2 * 1000 * 50 / 10), abs=0.01)


def test_eoq_nonpositive_returns_zero():
    assert economic_order_quantity(0, 50, 10) == 0
    assert economic_order_quantity(100, 50, 0) == 0


def test_days_of_supply():
    assert days_of_supply(100, 10) == 10.0
    assert days_of_supply(100, 0) == 100.0  # guard divide-by-zero


def test_restock_decision():
    needs, qty = restock_decision(50, 100, 10, coverage_days=7)
    assert needs is True
    assert qty == int(np.ceil((100 + 70) - 50))  # 120

    needs2, qty2 = restock_decision(150, 100, 10, coverage_days=7)
    assert needs2 is False
    assert qty2 == 0


def test_load_data_has_expected_products(data):
    assert {"P101", "P102", "P103", "P104", "P105"}.issubset(set(data["product_id"].unique()))


def test_forecast_demand_uses_model_file(data):
    model_file = "random_forest_model.pkl"
    if not os.path.exists(model_file):
        pytest.skip("random_forest_model.pkl not present")
    _, dates, preds = forecast_demand("P101", model_file=model_file, horizon=14, df=data)
    assert len(preds) == 14
    assert len(dates) == 14
    assert np.all(preds >= 0)  # clipped to non-negative


def test_summarize_optimization_schema(data):
    model_file = "random_forest_model.pkl"
    if not os.path.exists(model_file):
        pytest.skip("random_forest_model.pkl not present")
    opt = summarize_optimization("P101", df=data, model_file=model_file)
    for key in ["product_id", "avg_daily_demand", "safety_stock", "reorder_point",
                "eoq", "days_of_supply", "needs_restock", "recommended_order_qty"]:
        assert key in opt
    assert isinstance(opt["needs_restock"], bool)
