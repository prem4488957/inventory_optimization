# tests/test_recommendation_engine.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation_engine import (
    load_data,
    build_sales_pivot,
    product_similarity,
    recommend_bundles,
    bundle_sales_impact,
)


def test_load_data():
    df = load_data()
    assert df["product_id"].nunique() >= 5


def test_pivot_shape():
    df = load_data()
    pivot = build_sales_pivot(df)
    assert pivot.shape[1] == df["product_id"].nunique()
    assert pivot.shape[0] == df["date"].nunique()


def test_similarity_matrix_is_symmetric():
    df = load_data()
    corr = product_similarity(df)
    n = df["product_id"].nunique()
    assert corr.shape == (n, n)
    # diagonal is 1.0
    assert corr.iloc[0, 0] == pytest_approx(1.0)
    # symmetric
    assert corr.iloc[0, 1] == pytest_approx(corr.iloc[1, 0])


def test_recommend_bundles_returns_pairs():
    df = load_data()
    recs = recommend_bundles(df, top_n=5)
    assert len(recs) == 5
    for r in recs:
        assert "product_a" in r and "product_b" in r
        assert r["similarity"] >= -1.0 and r["similarity"] <= 1.0


def test_bundle_sales_impact_adds_daily_sales():
    df = load_data()
    recs = recommend_bundles(df, top_n=3)
    recs = bundle_sales_impact(df, recs)
    for r in recs:
        assert "bundle_avg_daily_sales" in r
        assert r["bundle_avg_daily_sales"] > 0


def pytest_approx(v):
    from pytest import approx
    return approx(v)
