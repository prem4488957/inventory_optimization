# competitor_intelligence.py
"""
Competitor Intelligence Engine
Provides competitor price benchmarking (Amazon, Flipkart, Reliance Digital),
Price Elasticity of Demand (PED) estimation, and Dynamic Market Threat Detection.
"""

import numpy as np
import pandas as pd

# Category-specific Price Elasticity of Demand (PED)
CATEGORY_ELASTICITY = {
    "Smartphone": -1.8,
    "Mobile": -1.8,
    "Appliance": -1.4,
    "Refrigerator": -1.3,
    "Air Conditioner": -1.5,
    "Television": -1.6,
    "Washing Machine": -1.4,
    "Kitchen Appliance": -1.2,
    "Air Purifier": -1.3,
    "Dishwasher": -1.2,
}


def generate_competitor_prices(product_id: str, our_price: float, seed: int = 42) -> dict:
    """
    Generates realistic competitor benchmark prices (Amazon, Flipkart, Reliance Digital)
    for a given product and unit price.
    """
    # Deterministic seed based on product_id hash
    pid_num = sum(ord(c) for c in product_id) + seed
    rng = np.random.RandomState(pid_num)

    # Competitor price variation between -10% and +8% relative to our price
    amazon_var = rng.uniform(-0.09, 0.06)
    flipkart_var = rng.uniform(-0.10, 0.07)
    reliance_var = rng.uniform(-0.07, 0.08)

    amazon_price = round(our_price * (1.0 + amazon_var), 2)
    flipkart_price = round(our_price * (1.0 + flipkart_var), 2)
    reliance_price = round(our_price * (1.0 + reliance_var), 2)

    comp_avg = round(float(np.mean([amazon_price, flipkart_price, reliance_price])), 2)
    rpi = round(our_price / comp_avg, 3) if comp_avg > 0 else 1.0

    return {
        "product_id": product_id,
        "our_price": our_price,
        "amazon_price": amazon_price,
        "flipkart_price": flipkart_price,
        "reliance_price": reliance_price,
        "competitor_avg_price": comp_avg,
        "relative_price_index": rpi,
    }


def analyze_market_threat(our_price: float, comp_metrics: dict, category: str = "Smartphone") -> dict:
    """
    Analyzes market competitive positioning, elasticity impacts, and threat recommendations.
    """
    rpi = comp_metrics["relative_price_index"]
    comp_avg = comp_metrics["competitor_avg_price"]
    ped = CATEGORY_ELASTICITY.get(category, -1.5)

    price_diff_pct = round(((our_price - comp_avg) / comp_avg) * 100.0, 2)

    # Estimate demand impact multiplier based on elasticity
    # % change in price * PED = % change in demand
    demand_impact_pct = round(-1.0 * price_diff_pct * abs(ped), 1)

    if price_diff_pct > 6.0:
        threat_level = "CRITICAL_UNDERPRICING_THREAT"
        threat_badge = "🚨 COMPETITOR UNDERPRICING RISK"
        recommendation = (
            f"Competitors are averaging {abs(price_diff_pct):.1f}% cheaper (₹{comp_avg:,.2f}). "
            f"Expected demand drag: {abs(demand_impact_pct):.1f}%. Recommend running a 5% promotional match."
        )
    elif price_diff_pct < -6.0:
        threat_level = "MARGIN_EXPANSION_OPPORTUNITY"
        threat_badge = "⚡ MARGIN EXPANSION OPPORTUNITY"
        recommendation = (
            f"Our price is {abs(price_diff_pct):.1f}% lower than market average (₹{comp_avg:,.2f}). "
            f"Expected demand surge: +{abs(demand_impact_pct):.1f}%. Maintain price or expand buffer stock."
        )
    else:
        threat_level = "MARKET_PARITY"
        threat_badge = "✅ MARKET PRICE PARITY"
        recommendation = (
            f"Our pricing is in optimal parity with competitor average (₹{comp_avg:,.2f}, variance {price_diff_pct:+.1f}%). "
            "Zero pricing threat detected."
        )

    return {
        "product_id": comp_metrics["product_id"],
        "category": category,
        "price_diff_pct": price_diff_pct,
        "ped": ped,
        "demand_impact_pct": demand_impact_pct,
        "threat_level": threat_level,
        "threat_badge": threat_badge,
        "recommendation": recommendation,
    }


def get_competitor_intelligence_summary(product_id: str, our_price: float, category: str = "Smartphone") -> dict:
    """
    Convenience orchestrator for a full product competitor intelligence summary.
    """
    comp_prices = generate_competitor_prices(product_id, our_price)
    threat_analysis = analyze_market_threat(our_price, comp_prices, category)
    return {**comp_prices, **threat_analysis}
