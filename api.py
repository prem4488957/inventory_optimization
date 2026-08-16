# api.py
# Stage 16 - Flask REST API wrapper around the forecasting + inventory optimization modules.
# Run with:  python api.py   (defaults to http://127.0.0.1:5000)
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import json

from inventory_optimizer import (
    load_engineered_data,
    summarize_optimization,
    forecast_demand,
)
import database as db
import recommendation_engine as rec

app = Flask(__name__)
CORS(app)

# Cache the engineered data and best model name at startup
DATA = load_engineered_data()
with open("model_metrics.json", "r") as f:
    METRICS = json.load(f)
BEST_MODEL = METRICS.get("best_model", "Random Forest")
# Best model -> pickle file mapping (only scalar-MAE models have pkl files)
MODEL_FILES = {
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "LightGBM": "lightgbm_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Linear Regression": "linear_regression_model.pkl",
    "Support Vector (RBF)": "support_vector_svr_model.pkl",
}


def _resolve_model(model_name):
    model_name = model_name or BEST_MODEL
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_FILES)}")
    return model_name, MODEL_FILES[model_name]


@app.route("/")
def home():
    return jsonify({
        "service": "Retail Inventory AI API",
        "version": "1.0",
        "best_model": BEST_MODEL,
        "endpoints": [
            "/api/health",
            "/api/products",
            "/api/forecast/<product_id>?model=...&horizon=14",
            "/api/optimize/<product_id>?model=...&stock=60&service_level=95",
            "/api/optimize/all?model=...",
            "/api/recommendations",
            "/api/metrics",
            "/api/db/<table>",
        ],
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "products": DATA["product_id"].nunique()})


@app.route("/api/products")
def products():
    cols = ["product_id", "product_name", "category", "lead_time_days"]
    return jsonify(DATA[cols].drop_duplicates().to_dict(orient="records"))


@app.route("/api/forecast/<product_id>")
def forecast(product_id):
    model_name = request.args.get("model")
    horizon = request.args.get("horizon", 14, type=int)
    try:
        model_name, model_file = _resolve_model(model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if product_id not in DATA["product_id"].unique():
        return jsonify({"error": f"Unknown product '{product_id}'"}), 404

    _, dates, preds = forecast_demand(product_id, model_file=model_file, horizon=horizon, df=DATA)
    return jsonify({
        "product_id": product_id,
        "model": model_name,
        "horizon_days": horizon,
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in dates],
        "forecast_units": [round(float(p), 2) for p in preds],
    })


@app.route("/api/optimize/<product_id>")
def optimize(product_id):
    model_name = request.args.get("model")
    stock = request.args.get("stock", 60, type=int)
    service_level = request.args.get("service_level", 95, type=int)
    try:
        model_name, model_file = _resolve_model(model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if product_id not in DATA["product_id"].unique():
        return jsonify({"error": f"Unknown product '{product_id}'"}), 404

    opt = summarize_optimization(
        product_id, df=DATA, model_file=model_file,
        current_stock=stock, service_level_pct=service_level,
    )
    opt["model"] = model_name
    db.save_optimization(opt, model_name)
    db.save_forecast(product_id, model_name, opt["forecast_dates"], opt["forecast"])
    return jsonify(opt)


@app.route("/api/optimize/all")
def optimize_all():
    model_name = request.args.get("model")
    stock = request.args.get("stock", 60, type=int)
    service_level = request.args.get("service_level", 95, type=int)
    try:
        model_name, model_file = _resolve_model(model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    results = []
    for pid in DATA["product_id"].unique():
        opt = summarize_optimization(
            pid, df=DATA, model_file=model_file,
            current_stock=stock, service_level_pct=service_level,
        )
        opt["model"] = model_name
        results.append(opt)
    return jsonify({"model": model_name, "results": results})


@app.route("/api/recommendations")
def recommendations():
    recs = rec.bundle_sales_impact(DATA)
    return jsonify(recs)


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.route("/api/db/<table>")
def db_table(table):
    if table not in {"products", "sales", "forecasts", "inventory_optimization"}:
        return jsonify({"error": f"Unknown table '{table}'"}), 400
    df = db.read_table(table)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/competitors/<product_id>")
def competitor_intelligence(product_id):
    from competitor_intelligence import get_competitor_intelligence_summary
    prod_df = DATA[DATA["product_id"] == product_id]
    if prod_df.empty:
        return jsonify({"error": f"Product '{product_id}' not found"}), 404
    unit_price = float(prod_df["unit_price"].iloc[-1])
    if unit_price < 2000:
        unit_price *= 83.0
    category = prod_df["category"].iloc[0]
    info = get_competitor_intelligence_summary(product_id, unit_price, category)
    return jsonify(info)


if __name__ == "__main__":
    print(f"[+] Retail Inventory AI API running with best model: {BEST_MODEL}")
    app.run(host="0.0.0.0", port=5000, debug=False)
