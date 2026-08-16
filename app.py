# app.py
import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from inventory_optimizer import (
    load_engineered_data,
    build_future_features,
    z_for_service_level,
    safety_stock,
    reorder_point,
    days_of_supply,
    restock_decision,
    DEFAULT_FEATURES,
)
from chatbot import process_query
from competitor_intelligence import get_competitor_intelligence_summary
from explainability import (
    prepare_sample,
    feature_importance_df,
    shap_values,
    generate_explanation,
    FEATURE_LABELS,
)

# Page Configuration
st.set_page_config(
    page_title="CYBER_INVENTORY // AI DEMAND FORECASTER",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# CYBERPUNK 2077 DESIGN SYSTEM — Futuristic High-Contrast Neon HUD CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800;900&family=JetBrains+Mono:wght@400;500;700;800&family=Rajdhani:wght@500;600;700&display=swap');

:root {
  --cyber-bg: #040408;
  --cyber-surface: #0A0A14;
  --cyber-card: #0F0F1D;
  --cyber-cyan: #00F0FF;
  --cyber-cyan-glow: rgba(0, 240, 255, 0.35);
  --cyber-magenta: #FF007F;
  --cyber-magenta-glow: rgba(255, 0, 127, 0.35);
  --cyber-yellow: #FFE600;
  --cyber-green: #00FF66;
  --cyber-red: #FF2E63;
  --cyber-purple: #9D4EDD;
  --cyber-text: #E0F7FA;
  --cyber-text-muted: #80DEEA;
  --cyber-text-dim: #455A64;
  --cyber-border-cyan: 1px solid #00F0FF;
  --cyber-border-magenta: 1px solid #FF007F;
  --cyber-border-dim: 1px solid rgba(0, 240, 255, 0.2);
}

/* Global Cyberpunk Reset */
html, body, [class*="css"] {
  font-family: 'Rajdhani', 'Orbitron', -apple-system, sans-serif;
  color: var(--cyber-text);
}

.stApp {
  background: 
    linear-gradient(180deg, rgba(4, 4, 8, 0.95) 0%, rgba(10, 10, 20, 0.98) 100%),
    radial-gradient(circle at 10% 20%, rgba(255, 0, 127, 0.08) 0%, transparent 40%),
    radial-gradient(circle at 90% 80%, rgba(0, 240, 255, 0.08) 0%, transparent 40%);
  background-attachment: fixed;
}

/* Scanline visual effect overlay */
.stApp::before {
  content: " ";
  display: block;
  position: fixed;
  top: 0; left: 0; bottom: 0; right: 0;
  background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.03), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.03));
  z-index: 9999;
  background-size: 100% 3px, 6px 100%;
  pointer-events: none;
  opacity: 0.6;
}

/* Main Container Padding */
.block-container {
  padding-top: 1.5rem;
  padding-bottom: 3rem;
  max-width: 1450px;
}

/* Cyberpunk Sidebar */
[data-testid="stSidebar"] {
  background-color: #06060C !important;
  border-right: 1px solid rgba(0, 240, 255, 0.3) !important;
  box-shadow: 5px 0 25px rgba(0, 240, 255, 0.15);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
  font-family: 'Orbitron', monospace;
  color: var(--cyber-cyan) !important;
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase;
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.6);
  border-bottom: 1px solid rgba(0, 240, 255, 0.3);
  padding-bottom: 0.4rem;
  margin-top: 1.2rem;
}

/* Cyber Inputs & Selectboxes */
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
  background: #0A0A14 !important;
  border: 1px solid var(--cyber-cyan) !important;
  box-shadow: 0 0 10px rgba(0, 240, 255, 0.15) !important;
  border-radius: 2px !important;
  color: var(--cyber-cyan) !important;
}

/* Sliders */
[data-baseweb="slider"] {
  margin-top: 0.5rem;
}

/* Buttons — Cyberpunk Cut Style */
.stButton > button, .stDownloadButton > button {
  font-family: 'Orbitron', sans-serif !important;
  background: linear-gradient(135deg, #FF007F 0%, #7B2CBF 100%) !important;
  color: #FFFFFF !important;
  border: 1px solid #00F0FF !important;
  border-radius: 0px !important;
  clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px)) !important;
  padding: 0.6rem 1.6rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  box-shadow: 0 0 15px rgba(255, 0, 127, 0.5) !important;
  transition: all 0.2s ease-in-out !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
  box-shadow: 0 0 25px rgba(0, 240, 255, 0.8) !important;
  transform: scale(1.02) !important;
}

/* Cyber HUD Hero Header */
.cyber-hero {
  background: linear-gradient(135deg, rgba(15, 15, 29, 0.95) 0%, rgba(6, 6, 12, 0.98) 100%);
  border: 1px solid var(--cyber-cyan);
  box-shadow: 0 0 25px rgba(0, 240, 255, 0.25), inset 0 0 15px rgba(0, 240, 255, 0.1);
  padding: 1.8rem 2.2rem;
  margin-bottom: 1.8rem;
  position: relative;
  overflow: hidden;
  clip-path: polygon(0 0, calc(100% - 20px) 0, 100% 20px, 100% 100%, 20px 100%, 0 calc(100% - 20px));
}

.cyber-hero::after {
  content: '// SYSTEM_READY';
  position: absolute;
  top: 12px;
  right: 25px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--cyber-cyan);
  letter-spacing: 0.1em;
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.8);
}

.cyber-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: rgba(255, 0, 127, 0.15);
  border: 1px solid #FF007F;
  color: #FF007F;
  font-family: 'Orbitron', monospace;
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.3rem 0.8rem;
  letter-spacing: 0.1em;
  text-shadow: 0 0 8px rgba(255, 0, 127, 0.6);
  margin-bottom: 0.6rem;
}

.cyber-title {
  font-family: 'Orbitron', sans-serif;
  font-size: 2.2rem;
  font-weight: 900;
  letter-spacing: 0.02em;
  color: #FFFFFF;
  text-shadow: 0 0 12px rgba(0, 240, 255, 0.8), 0 0 24px rgba(255, 0, 127, 0.4);
  margin: 0 0 0.4rem 0;
  text-transform: uppercase;
}

.cyber-subtitle {
  font-family: 'Rajdhani', sans-serif;
  color: var(--cyber-text-muted);
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0;
  letter-spacing: 0.04em;
}

/* Cyber KPI Cards Grid */
.cyber-kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1rem;
  margin-bottom: 1.6rem;
}

.cyber-kpi-card {
  background: rgba(15, 15, 29, 0.9);
  border: 1px solid var(--cyber-cyan);
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.15);
  padding: 1.1rem 1.2rem;
  position: relative;
  clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
  transition: all 0.2s ease;
}

.cyber-kpi-card:hover {
  border-color: #FF007F;
  box-shadow: 0 0 25px rgba(255, 0, 127, 0.35);
  transform: translateY(-2px);
}

.cyber-kpi-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.4rem;
}

.cyber-kpi-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--cyber-yellow);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.cyber-kpi-label {
  font-family: 'Orbitron', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--cyber-cyan);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}

.cyber-kpi-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.65rem;
  font-weight: 800;
  color: #FFFFFF;
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.6);
  letter-spacing: -0.02em;
}

.cyber-kpi-sub {
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.78rem;
  color: var(--cyber-text-muted);
  margin-top: 0.25rem;
  font-weight: 600;
}

/* Cyber Status Alerts */
.cyber-alert-danger {
  background: linear-gradient(135deg, rgba(255, 0, 127, 0.2) 0%, rgba(50, 0, 25, 0.5) 100%);
  border: 1px solid #FF007F;
  box-shadow: 0 0 25px rgba(255, 0, 127, 0.4);
  padding: 1.3rem 1.6rem;
  margin: 1.5rem 0;
  clip-path: polygon(0 0, calc(100% - 15px) 0, 100% 15px, 100% 100%, 15px 100%, 0 calc(100% - 15px));
  display: flex;
  align-items: center;
  gap: 1.2rem;
}

.cyber-alert-success {
  background: linear-gradient(135deg, rgba(0, 255, 102, 0.15) 0%, rgba(0, 50, 20, 0.4) 100%);
  border: 1px solid #00FF66;
  box-shadow: 0 0 25px rgba(0, 255, 102, 0.3);
  padding: 1.3rem 1.6rem;
  margin: 1.5rem 0;
  clip-path: polygon(0 0, calc(100% - 15px) 0, 100% 15px, 100% 100%, 15px 100%, 0 calc(100% - 15px));
  display: flex;
  align-items: center;
  gap: 1.2rem;
}

.cyber-alert-title {
  font-family: 'Orbitron', monospace;
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  margin-bottom: 0.3rem;
  text-shadow: 0 0 8px currentColor;
}

/* Streamlit Container Override */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(15, 15, 29, 0.85) !important;
  border: 1px solid rgba(0, 240, 255, 0.3) !important;
  box-shadow: 0 0 20px rgba(0, 240, 255, 0.1) !important;
  clip-path: polygon(0 0, calc(100% - 15px) 0, 100% 15px, 100% 100%, 15px 100%, 0 calc(100% - 15px));
  padding: 1.3rem !important;
}

/* Sidebar Info Card */
.cyber-sidebar-box {
  background: #0A0A14;
  border: 1px solid var(--cyber-cyan);
  box-shadow: inset 0 0 10px rgba(0, 240, 255, 0.1);
  padding: 0.85rem 1rem;
  margin: 1rem 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
}

.cyber-sidebar-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.4rem;
}
.cyber-sidebar-row:last-child { margin-bottom: 0; }
.cyber-sidebar-label { color: var(--cyber-text-muted); }
.cyber-sidebar-val { font-weight: 700; color: var(--cyber-yellow); }

/* Table overrides */
[data-testid="stDataFrame"] {
  border: 1px solid var(--cyber-cyan) !important;
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.2) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #040408; }
::-webkit-scrollbar-thumb { background: #00F0FF; box-shadow: 0 0 10px #00F0FF; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data & Model Registry Initialization (UNTOUCHED LOGIC)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    "XGBoost (Best)": os.path.join(BASE_DIR, "xgboost_model.pkl"),
    "Random Forest": os.path.join(BASE_DIR, "random_forest_model.pkl"),
    "Gradient Boosting": os.path.join(BASE_DIR, "gradient_boosting_model.pkl"),
    "LightGBM": os.path.join(BASE_DIR, "lightgbm_model.pkl"),
    "Linear Regression": os.path.join(BASE_DIR, "linear_regression_model.pkl"),
    "Support Vector (RBF)": os.path.join(BASE_DIR, "support_vector_svr_model.pkl"),
}

@st.cache_data(ttl=1)
def load_resources():
    csv_path = os.path.join(BASE_DIR, "featured_sales_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = load_engineered_data()
    df["date"] = pd.to_datetime(df["date"])
    models = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    metrics = None
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        pass
    return df, models, metrics

df, models, metrics = load_resources()

# ---------------------------------------------------------------------------
# Cyberpunk Sidebar: Neural Control Hub
# ---------------------------------------------------------------------------
st.sidebar.markdown("### // TARGET_SELECTION")

brands = sorted(df["brand"].unique())
selected_brand = st.sidebar.selectbox("Brand Matrix:", brands)

brand_products = sorted(df[df["brand"] == selected_brand]["product_name"].unique())
selected_product = st.sidebar.selectbox("Product Model:", brand_products)

st.sidebar.markdown("### // NEURAL_ENGINE")

selected_model = st.sidebar.selectbox(
    "Architecture Core:",
    list(MODEL_FILES.keys()),
    index=0,
    help="Select the trained neural/tree model for demand prediction.",
)
model = models[selected_model]

show_xai = st.sidebar.checkbox("🧠 Neural XAI (SHAP)", value=True,
                               help="Enable SHAP explainability engine.")

st.sidebar.markdown("### // SYSTEM_POLICY")

horizon = st.sidebar.slider("Forecast Horizon (Days):", min_value=7, max_value=60, value=30,
                            help="Prediction window length.")

prod_df = df[(df["brand"] == selected_brand) & (df["product_name"] == selected_product)] \
    .sort_values("date").reset_index(drop=True)
product_id = prod_df["product_id"].iloc[0]
lead_time = int(prod_df["lead_time_days"].iloc[0])
unit_price = float(prod_df["unit_price"].iloc[-1])
if unit_price < 2000:
    unit_price *= 83.0  # Fail-safe USD -> INR conversion
category = prod_df["category"].iloc[0]

# Cyberpunk Sidebar Info HUD
st.sidebar.markdown(f"""
<div class="cyber-sidebar-box">
  <div class="cyber-sidebar-row"><span class="cyber-sidebar-label">CATEGORY:</span><span class="cyber-sidebar-val">{category.upper()}</span></div>
  <div class="cyber-sidebar-row"><span class="cyber-sidebar-label">PROD_ID:</span><span class="cyber-sidebar-val">{product_id}</span></div>
  <div class="cyber-sidebar-row"><span class="cyber-sidebar-label">LEAD_TIME:</span><span class="cyber-sidebar-val">{lead_time} DAYS</span></div>
  <div class="cyber-sidebar-row"><span class="cyber-sidebar-label">UNIT_VAL:</span><span class="cyber-sidebar-val">₹{unit_price:,.2f}</span></div>
</div>
""", unsafe_allow_html=True)

current_stock = st.sidebar.number_input("Warehouse Stock (Units):", min_value=0, value=60, step=5)
service_level_target = st.sidebar.slider("Service Target Level (%):", min_value=90, max_value=99, value=95)



# ---------------------------------------------------------------------------
# Prediction & Inventory Calculations (100% UNTOUCHED BACKEND LOGIC)
# ---------------------------------------------------------------------------
future_features_df, future_dates = build_future_features(prod_df, horizon=horizon)
preds = model.predict(future_features_df)
forecasted_daily_units = np.clip(preds, 0, None)

avg_forecast_daily = float(np.mean(forecasted_daily_units))
forecast_std = float(np.std(forecasted_daily_units)) if float(np.std(forecasted_daily_units)) > 0 else 2.0

month_forecast = forecasted_daily_units[:30]
month_total = int(month_forecast.sum())
month_avg = float(month_forecast.mean())

z = z_for_service_level(service_level_target)
ss = safety_stock(forecast_std, lead_time, service_level_target)
rop = reorder_point(avg_forecast_daily, lead_time, forecast_std, service_level_target)
dos = days_of_supply(current_stock, avg_forecast_daily)
needs_restock, order_qty = restock_decision(current_stock, rop, avg_forecast_daily)

# ---------------------------------------------------------------------------
# Cyberpunk Sidebar: Neural AI Chatbot Assistant
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### // NEURAL_AI_ASSISTANT")

with st.sidebar.expander("💬 CYBER AI CHATBOT", expanded=True):
    st.markdown("<p style='color:#80DEEA;font-family:JetBrains Mono;font-size:0.8rem;'>Query telemetry, ROP, stock risks & recommendations.</p>", unsafe_allow_html=True)
    
    chat_box = st.container(height=280)
    with chat_box:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
            
        for q, a in st.session_state["chat_history"]:
            st.chat_message("user").write(q)
            st.chat_message("assistant").markdown(a)
            
    if st.session_state.get("chat_history"):
        if st.button("🗑️ Clear Chat History"):
            st.session_state["chat_history"] = []
            st.rerun()

    user_input = st.sidebar.chat_input("Ask inventory AI...")
    if user_input:
        response_text = process_query(user_input, df, metrics)
        st.session_state["chat_history"].append((user_input, response_text))
        st.rerun()

# ---------------------------------------------------------------------------
# Cyberpunk HUD Hero Banner (Full Width — Clean Visual Hierarchy)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="cyber-hero">
  <div class="cyber-badge">⚡ NEURAL ENGINE // HYPER-FORECAST ACTIVE</div>
  <h1 class="cyber-title">📱 AI Demand Forecasting & Inventory Optimization Platform</h1>
  <p class="cyber-subtitle">Automated restock intelligence for Smartphones & Electrical Appliances.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cyber Matrix KPI Cards Grid (Full 100% Width — Zero Text Overlap)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="cyber-kpi-grid">
  <div class="cyber-kpi-card">
    <div class="cyber-kpi-header">
      <div class="cyber-kpi-label">Month Demand</div>
      <div class="cyber-kpi-tag">[ 30D.SUM ]</div>
    </div>
    <div class="cyber-kpi-value">{month_total:,}</div>
    <div class="cyber-kpi-sub">Units aggregate forecast</div>
  </div>
  <div class="cyber-kpi-card">
    <div class="cyber-kpi-header">
      <div class="cyber-kpi-label">Daily Demand</div>
      <div class="cyber-kpi-tag">[ VELOCITY ]</div>
    </div>
    <div class="cyber-kpi-value">{avg_forecast_daily:.1f}</div>
    <div class="cyber-kpi-sub">Units mean / day</div>
  </div>
  <div class="cyber-kpi-card">
    <div class="cyber-kpi-header">
      <div class="cyber-kpi-label">Safety Stock</div>
      <div class="cyber-kpi-tag">[ BUFFER ]</div>
    </div>
    <div class="cyber-kpi-value" style="color:var(--cyber-cyan)">{int(np.ceil(ss)):,}</div>
    <div class="cyber-kpi-sub">{service_level_target}% target protection</div>
  </div>
  <div class="cyber-kpi-card">
    <div class="cyber-kpi-header">
      <div class="cyber-kpi-label">Reorder Point</div>
      <div class="cyber-kpi-tag">[ ROP.SYS ]</div>
    </div>
    <div class="cyber-kpi-value" style="color:var(--cyber-yellow)">{int(np.ceil(rop)):,}</div>
    <div class="cyber-kpi-sub">{lead_time}-day lead threshold</div>
  </div>
  <div class="cyber-kpi-card">
    <div class="cyber-kpi-header">
      <div class="cyber-kpi-label">Days Supply</div>
      <div class="cyber-kpi-tag">[ RUNWAY ]</div>
    </div>
    <div class="cyber-kpi-value" style="color:var(--cyber-magenta)">{dos:.1f}</div>
    <div class="cyber-kpi-sub">Current stock: {current_stock}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cyber Alert Banners
# ---------------------------------------------------------------------------
if needs_restock:
    st.markdown(f"""
    <div class="cyber-alert-danger">
      <div style="font-size:2rem;color:#FF007F;text-shadow:0 0 12px #FF007F">⚠️</div>
      <div>
        <div class="cyber-alert-title" style="color:#FF007F">// CRITICAL ALERT: RESTOCK TRIGGERED</div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:1rem;color:#FFFFFF;font-weight:600">
          Warehouse stock level (<strong>{current_stock} units</strong>) breached Reorder Point threshold (<strong>{int(np.ceil(rop))} units</strong>).
          Execute immediate Purchase Order for <strong>{order_qty} units</strong>.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="cyber-alert-success">
      <div style="font-size:2rem;color:#00FF66;text-shadow:0 0 12px #00FF66">⚡</div>
      <div>
        <div class="cyber-alert-title" style="color:#00FF66">// SYSTEM STATUS: OPTIMAL INVENTORY</div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:1rem;color:#FFFFFF;font-weight:600">
          Current warehouse inventory (<strong>{current_stock} units</strong>) safely exceeds Reorder Point threshold (<strong>{int(np.ceil(rop))} units</strong>). Zero deficit.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cyberpunk Demand Trajectory Plot (Plotly)
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(f"<h3 style='font-family:Orbitron;color:var(--cyber-cyan);letter-spacing:0.05em;'>// TELEMETRY: DEMAND TRAJECTORY & {horizon}-DAY FORECAST</h3>", unsafe_allow_html=True)
    
    fig = go.Figure()
    recent_history = prod_df.tail(60)
    
    # Historical Sales Trace (Neon Cyan)
    fig.add_trace(go.Scatter(
        x=recent_history["date"],
        y=recent_history["units_sold"],
        mode="lines",
        name="HISTORICAL SALES",
        line=dict(color="#00F0FF", width=2.8, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(0, 240, 255, 0.12)"
    ))
    

    # AI Forecast Trace (Neon Magenta)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecasted_daily_units,
        mode="lines+markers",
        name=f"AI FORECAST ({selected_model.upper()})",
        line=dict(color="#FF007F", width=2.8, dash="dash", shape="spline"),
        marker=dict(size=7, color="#FF007F", symbol="diamond"),
        fill="tozeroy",
        fillcolor="rgba(255, 0, 127, 0.12)"
    ))
    
    # ROP Threshold Line (Neon Yellow)
    fig.add_shape(
        type="line",
        x0=recent_history["date"].iloc[0],
        y0=rop,
        x1=future_dates[-1],
        y1=rop,
        line=dict(color="#FFE600", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=recent_history["date"].iloc[3],
        y=rop,
        text=f"REORDER THRESHOLD ({int(np.ceil(rop))} units)",
        showarrow=False,
        xanchor="left",
        yshift=-18,
        font=dict(color="#FFE600", size=11, family="Orbitron")
    )
    
    fig.update_layout(
        title=dict(
            text=f"<b>{selected_brand.upper()} {selected_product.upper()}</b> // DEMAND MATRIX TELEMETRY",
            font=dict(family="Orbitron", size=15, color="#00F0FF")
        ),
        xaxis_title="DATE",
        yaxis_title="UNITS / DEMAND",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 7, 15, 0.9)",
        font=dict(family="JetBrains Mono", color="#80DEEA"),
        xaxis=dict(
            gridcolor="rgba(0, 240, 255, 0.15)",
            tickfont=dict(color="#80DEEA"),
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="rgba(0, 240, 255, 0.15)",
            tickfont=dict(color="#80DEEA"),
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="left",
            x=0,
            font=dict(color="#00F0FF", size=11, family="Orbitron")
        ),
        margin=dict(l=20, r=20, t=75, b=20),
        height=460
    )
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Cyberpunk Competitor Intelligence & Price Elasticity Matrix
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("<h3 style='font-family:Orbitron;color:var(--cyber-cyan);letter-spacing:0.05em;'>📡 COMPETITOR INTELLIGENCE & PRICE ELASTICITY MATRIX</h3>", unsafe_allow_html=True)
    comp_info = get_competitor_intelligence_summary(product_id, unit_price, category)
    
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    c1.metric("Amazon Price", f"₹{comp_info['amazon_price']:,.2f}")
    c2.metric("Flipkart Price", f"₹{comp_info['flipkart_price']:,.2f}")
    c3.metric("Reliance Digital", f"₹{comp_info['reliance_price']:,.2f}")
    c4.metric("Market Variance", f"{comp_info['price_diff_pct']:+.1f}%", delta=f"PED: {comp_info['ped']}")

    st.markdown(f"""
    <div style="background:rgba(255, 230, 0, 0.08);border:1px solid #FFE600;padding:0.9rem;margin-top:1rem;color:#E0F7FA;font-family:'Rajdhani',sans-serif;font-size:1rem;">
      <strong>{comp_info['threat_badge']}</strong>: {comp_info['recommendation']}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Enterprise XAI (SHAP) Visualizations
# ---------------------------------------------------------------------------
if show_xai:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<h3 style='font-family:Orbitron;color:var(--cyber-cyan);letter-spacing:0.05em;'>// NEURAL XAI: SHAP FEATURE ATTRIBUTION</h3>", unsafe_allow_html=True)

        try:
            model_key = MODEL_FILES[selected_model]
            with st.spinner("Decoding neural weights..."):
                if "xai_imp" not in st.session_state or st.session_state.get("xai_model") != model_key:
                    X_sample = prepare_sample(df)
                    explainer, sv = shap_values(model, X_sample)
                    imp_df = feature_importance_df(model, X_sample)
                    st.session_state["xai_model"] = model_key
                    st.session_state["xai_imp"] = imp_df
                    st.session_state["xai_explainer"] = explainer
                    st.session_state["xai_sv"] = sv
                    st.session_state["xai_sample"] = X_sample
                else:
                    imp_df = st.session_state["xai_imp"]
                    explainer = st.session_state["xai_explainer"]
                    sv = st.session_state["xai_sv"]
                    X_sample = st.session_state["xai_sample"]

            local_row = future_features_df.iloc[[0]]
            local_sv = explainer.shap_values(local_row[FEATURE_LABELS.keys()])
            local_feats = local_row.rename(columns=FEATURE_LABELS)

            explanation_text = generate_explanation(
                f"{selected_brand} {selected_product}",
                explainer, local_sv, local_feats, imp_df, top_n=3,
            )
            
            st.markdown(f"""
            <div style="background:rgba(0, 240, 255, 0.08);border:1px solid #00F0FF;box-shadow:0 0 15px rgba(0, 240, 255, 0.2);padding:1.1rem;margin-bottom:1.5rem;color:#E0F7FA;font-family:'Rajdhani',sans-serif;font-size:1rem;line-height:1.6;">
              ⚡ <strong>NEURAL DIAGNOSTIC INSIGHT:</strong> {explanation_text}
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1])

            # Matplotlib Cyberpunk Styling
            plt.style.use('dark_background')

            with c1:
                st.markdown("#### // FEATURE IMPORTANCE RANKING")
                st.dataframe(imp_df.head(10).set_index("Feature"), width="stretch")

                st.markdown("#### // LOCAL WATERFALL ATTRIBUTION")
                import shap as _shap
                exp = _shap.Explanation(
                    values=local_sv[0],
                    base_values=float(explainer.expected_value),
                    data=local_feats.iloc[0].values,
                    feature_names=list(local_feats.columns),
                )
                fig_local, ax_local = plt.subplots(figsize=(6, 4))
                fig_local.patch.set_facecolor('#0A0A14')
                ax_local.set_facecolor('#0A0A14')
                _shap.waterfall_plot(exp, show=False, max_display=8)
                st.pyplot(fig_local, width="stretch")
                plt.close(fig_local)

            with c2:
                st.markdown("#### // SHAP GLOBAL ATTRIBUTION MATRIX")
                fig_sum, ax_sum = plt.subplots(figsize=(6, 4))
                fig_sum.patch.set_facecolor('#0A0A14')
                ax_sum.set_facecolor('#0A0A14')
                _shap.summary_plot(sv, X_sample.head(len(sv)).rename(columns=FEATURE_LABELS),
                                   max_display=10, show=False)
                st.pyplot(fig_sum, width="stretch")
                plt.close(fig_sum)
                st.caption("Magenta = Higher value, Cyan = Lower value. Position right/left = pushes forecast up/down.")
        except Exception as e:
            st.warning(f"XAI attribution unavailable for core {selected_model}: {e}")

# ---------------------------------------------------------------------------
# Cyber Leaderboard & Accuracy Table (Placed at Bottom Below XAI & Graphs)
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("<h3 style='font-family:Orbitron;color:var(--cyber-cyan);letter-spacing:0.05em;'>📊 Model Accuracy Comparison</h3>", unsafe_allow_html=True)
    if metrics:
        rows = []
        best = metrics["best_model"]
        for name, m in metrics["metrics"].items():
            if not isinstance(m, dict) or "MAE" not in m or not isinstance(m["MAE"], (int, float)):
                continue
            rows.append({
                "Architecture Core": name,
                "MAE (Units)": round(m["MAE"], 3),
                "RMSE (Units)": round(m["RMSE"], 3),
                "Neural Status": "👑 BEST OPTIMAL" if name == best else ("⚡ ACTIVE" if name == selected_model else "BENCHMARK"),
            })
        metrics_df = pd.DataFrame(rows).sort_values("MAE (Units)")
        st.dataframe(metrics_df.set_index("Architecture Core"), width="stretch")
        st.caption(f"Evaluated on test set split (from {metrics.get('split_date', '2025-10-01')}). "
                   f"Top architecture: **{best}**. Currently deployed core: **{selected_model}**.")
    else:
        st.info("Model metrics report missing. Execute `python train_model.py` to regenerate.")
