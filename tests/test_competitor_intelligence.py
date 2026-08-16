# tests/test_competitor_intelligence.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from competitor_intelligence import (
    generate_competitor_prices,
    analyze_market_threat,
    get_competitor_intelligence_summary,
)


def test_generate_competitor_prices():
    res = generate_competitor_prices("P101", 82917.0)
    assert res["product_id"] == "P101"
    assert res["our_price"] == 82917.0
    assert "amazon_price" in res and "flipkart_price" in res and "reliance_price" in res
    assert res["competitor_avg_price"] > 0
    assert res["relative_price_index"] > 0


def test_analyze_market_threat():
    comp_prices = generate_competitor_prices("P101", 82917.0)
    analysis = analyze_market_threat(82917.0, comp_prices, "Smartphone")
    assert "threat_level" in analysis
    assert "threat_badge" in analysis
    assert "recommendation" in analysis
    assert "demand_impact_pct" in analysis


def test_get_competitor_intelligence_summary():
    summary = get_competitor_intelligence_summary("P102", 66317.0, "Smartphone")
    assert summary["product_id"] == "P102"
    assert summary["our_price"] == 66317.0
    assert summary["amazon_price"] > 0
    assert summary["flipkart_price"] > 0
