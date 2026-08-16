# recommendation_engine.py
# Stage 14 - Product recommendation / bundling engine.
# Computes pairwise similarity between products based on their daily sales patterns
# and recommends bundling opportunities (complementary/highly-correlated products).
import pandas as pd
import numpy as np
import itertools


def load_data(path="featured_sales_data.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_sales_pivot(df):
    """Pivot daily sales into a product x date matrix (products as columns)."""
    pivot = df.pivot_table(index="date", columns="product_id", values="units_sold", aggfunc="mean")
    pivot = pivot.sort_index()
    return pivot.fillna(pivot.mean())


def product_similarity(df=None, method="pearson"):
    """Return a DataFrame of pairwise similarity scores between products."""
    if df is None:
        df = load_data()
    pivot = build_sales_pivot(df)

    if method == "pearson":
        corr = pivot.corr()
    else:  # spearman rank correlation
        corr = pivot.rank().corr()

    return corr


def recommend_bundles(df=None, top_n=5, min_similarity=0.0):
    """
    Rank all product pairs by sales-pattern similarity and return the best
    bundling recommendations. Returns a list of dicts.
    """
    if df is None:
        df = load_data()
    corr = product_similarity(df)

    name_map = df[["product_id", "product_name", "category"]].drop_duplicates().set_index("product_id")
    name_map["product_name"] = name_map["product_name"].astype(str)

    pairs = []
    for a, b in itertools.combinations(corr.columns, 2):
        score = corr.loc[a, b]
        if np.isnan(score):
            continue
        pairs.append({
            "product_a": a,
            "product_b": b,
            "name_a": name_map.loc[a, "product_name"],
            "name_b": name_map.loc[b, "product_name"],
            "category_a": name_map.loc[a, "category"],
            "category_b": name_map.loc[b, "category"],
            "similarity": round(float(score), 3),
        })

    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    pairs = [p for p in pairs if p["similarity"] >= min_similarity]
    return pairs[:top_n]


def bundle_sales_impact(df=None, pairs=None):
    """
    For each recommended bundle, estimate the joint daily sales (sum of both
    products' average daily sales) to gauge bundle size/priority.
    """
    if df is None:
        df = load_data()
    if pairs is None:
        pairs = recommend_bundles(df)

    avg_daily = df.groupby("product_id")["units_sold"].mean()
    for p in pairs:
        p["avg_daily_sales_a"] = round(float(avg_daily.get(p["product_a"], 0)), 2)
        p["avg_daily_sales_b"] = round(float(avg_daily.get(p["product_b"], 0)), 2)
        p["bundle_avg_daily_sales"] = round(p["avg_daily_sales_a"] + p["avg_daily_sales_b"], 2)
    return pairs


if __name__ == "__main__":
    data = load_data()
    print("=== Product sales-pattern correlation matrix ===")
    print(product_similarity(data).round(3).to_string())
    print("\n=== Top bundling recommendations ===")
    recs = bundle_sales_impact(data)
    for r in recs:
        print(f"{r['name_a']} + {r['name_b']}  | similarity={r['similarity']:.3f} "
              f"| bundle avg daily={r['bundle_avg_daily_sales']:.1f} units")
