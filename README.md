# AI Demand Forecasting & Inventory Optimization Platform

An end-to-end retail inventory intelligence system that forecasts daily product demand,
computes data-driven inventory policies (safety stock, reorder point, EOQ), recommends
product bundles, and exposes everything through a Streamlit dashboard, a Flask REST API,
and a SQLite persistence layer.

Built for a 5-product catalog (mobile phones + electrical appliances) with 2 years of
daily sales data (2024-01-01 to 2025-12-31).

---

## Catalog

| ID | Product | Category | Base demand | Lead time |
|----|---------|----------|-------------|-----------|
| P101 | iPhone 15 Pro | Mobile | 25/day | 4 days |
| P102 | Samsung Galaxy S24 | Mobile | 20/day | 3 days |
| P103 | Smart Refrigerator 450L | Appliance | 8/day | 7 days |
| P104 | Inverter Air Conditioner 1.5T | Appliance | 12/day | 6 days |
| P105 | 4K Smart OLED TV 55" | Appliance | 10/day | 5 days |

---

## Pipeline (20 Stages)

| # | Stage | File | Status |
|---|-------|------|--------|
| 1 | Dataset generation | `generate_data.py` | ✔ |
| 2 | Data cleaning | `generate_data.py` / `feature_engineering.py` | ✔ |
| 3 | EDA | `feature_engineering.py` (calendar + lag/rolling features) | ✔ |
| 4 | Feature engineering | `feature_engineering.py` | ✔ |
| 5 | Baseline forecasting | `train_model.py` (Linear, SVR) | ✔ |
| 6 | Seasonal baseline (Prophet-style) | `seasonal_model.py` (Holt-Winters) | ✔ |
| 7 | XGBoost | `train_model.py` | ✔ |
| 8 | LSTM (deep learning) | `deep_learning_model.py` (PyTorch) | ✔ |
| 9 | Model comparison | `model_metrics.json` | ✔ |
| 10 | Demand forecasting | `inventory_optimizer.py` | ✔ |
| 11 | Safety stock | `inventory_optimizer.py` | ✔ |
| 12 | Reorder point (ROP) | `inventory_optimizer.py` | ✔ |
| 13 | EOQ | `inventory_optimizer.py` | ✔ |
| 14 | Recommendation engine | `recommendation_engine.py` | ✔ |
| 15 | Database | `database.py` (SQLite) | ✔ |
| 16 | Flask API | `api.py` | ✔ |
| 17 | Dashboard | `app.py` (Streamlit) | ✔ |
| 18 | Testing | `tests/` (pytest) | ✔ |
| 19 | Deployment config | `requirements.txt`, `run_all.py` | ✔ |
| 20 | Documentation | `README.md` | ✔ |

> **Note on Stage 6:** Meta's Prophet requires the CmdStan C++ toolchain, which is not
> reliably installable on Windows/Python 3.13. Its analytical role (a traditional,
> interpretable seasonal time-series baseline to compare against the ML models) is filled
> by a statsmodels **multiplicative Holt-Winters** exponential smoothing model with weekly
> seasonality.

---

## Model Comparison

Evaluated on a hold-out test set (from `2025-10-01`, 460 samples) using a time-based split.
Per-product LSTM and Holt-Winters models are also trained.

| Model | MAE (units) | RMSE (units) |
|-------|-------------|--------------|
| Random Forest | 4.013 | 5.739 |
| XGBoost | 4.023 | 5.942 |
| LightGBM | 4.098 | 5.987 |
| Gradient Boosting | 4.134 | 6.040 |
| Linear Regression | 4.153 | 6.088 |
| LSTM (PyTorch) | 4.959 | 7.669 |
| Seasonal Baseline (Holt-Winters) | 6.362 | 9.597 |
| Support Vector (RBF) | 6.793 | 10.805 |

**Best model: Random Forest.**

---

## Installation

```bash
cd retail_inventory_ai
pip install -r requirements.txt
```

Python 3.13 recommended.

---

## Usage

### 1. Full pipeline (regenerate everything)

```bash
python run_all.py
```

This runs data generation → feature engineering → all model training → database seeding
and prints the final model comparison table.

### 2. Run individual stages

```bash
python generate_data.py        # Stage 1-2: raw dataset
python feature_engineering.py  # Stage 3-4: features
python train_model.py          # Stage 5/7/9: ML models + metrics
python seasonal_model.py       # Stage 6: seasonal baseline
python deep_learning_model.py  # Stage 8: LSTM
python database.py             # Stage 15: seed SQLite
```

### 3. Dashboard

```bash
streamlit run app.py
```

Features: product selector, model selector, service-level slider, live 14-day forecast,
safety stock / ROP / days-of-supply KPIs, restock banner, and the model accuracy table.

### 4. REST API

```bash
python api.py        # http://127.0.0.1:5000
```

Endpoints:

- `GET /api/health`
- `GET /api/products`
- `GET /api/forecast/<product_id>?model=Random+Forest&horizon=14`
- `GET /api/optimize/<product_id>?model=...&stock=60&service_level=95`
- `GET /api/optimize/all`
- `GET /api/recommendations`
- `GET /api/metrics`
- `GET /api/db/<table>`  (products | sales | forecasts | inventory_optimization)

### 5. Tests

```bash
python -m pytest tests -q
```

### 6. Reusable library

```python
from inventory_optimizer import summarize_optimization, load_engineered_data

df = load_engineered_data()
opt = summarize_optimization("P101", df=df, model_file="random_forest_model.pkl",
                             current_stock=60, service_level_pct=95)
print(opt["safety_stock"], opt["reorder_point"], opt["eoq"])
```

---

## Inventory Optimization Formulas

- **Safety stock** = `z × σ_daily × √(lead time)`, where `z` is the standard-normal
  quantile for the target service level (90%→1.28, 95%→1.65, 98%→2.05, 99%→2.33).
- **Reorder point (ROP)** = `(avg daily demand × lead time) + safety stock`.
- **EOQ** = `√(2 × annual demand × ordering cost / holding cost per unit per year)`.
- **Days of supply** = `current stock / avg daily forecast demand`.

---

## Project Structure

```
retail_inventory_ai/
├── generate_data.py            # Dataset generation
├── feature_engineering.py      # Feature engineering
├── train_model.py              # ML models (LightGBM, RF, GB, Linear, SVR, XGBoost)
├── seasonal_model.py           # Holt-Winters seasonal baseline
├── deep_learning_model.py      # PyTorch LSTM
├── inventory_optimizer.py      # Demand forecast + SS/ROP/EOQ functions
├── recommendation_engine.py    # Bundle recommendations
├── database.py                 # SQLite persistence
├── api.py                      # Flask REST API
├── app.py                      # Streamlit dashboard
├── run_all.py                  # Full pipeline orchestrator
├── requirements.txt
├── model_metrics.json          # Model comparison report
├── retail_sales_data.csv       # Raw dataset
├── featured_sales_data.csv     # Engineered dataset
├── *.pkl                       # Serialized trained models
├── inventory.db                # SQLite database
└── tests/                      # pytest test suite
```

---

## Research Paper / Presentation

The pipeline and results above are intended for a research paper and slide deck covering:
demand forecasting methodology comparison (ML vs. deep learning vs. traditional
time-series), inventory policy derivation, and a practical REST/dashboard deployment.
