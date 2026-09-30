# chatbot.py
"""
Cyber AI Inventory Assistant Engine
Provides 100% offline, instant natural language querying over sales telemetry,
inventory optimization policies (Safety Stock, ROP, EOQ), bundle recommendations,
and model performance metrics.
"""

import re
import numpy as np
import pandas as pd
from inventory_optimizer import (
    safety_stock,
    reorder_point,
    days_of_supply,
    restock_decision,
    economic_order_quantity,
)
from recommendation_engine import recommend_bundles, product_similarity


def process_query(query: str, df: pd.DataFrame, metrics: dict = None,
                  selected_brand: str = None, selected_product: str = None,
                  current_stock: int = 60) -> str:
    """
    Process a natural language user query over the inventory dataset and returns
    a formatted Cyberpunk Markdown response.
    """
    query_clean = query.strip().lower()
    
    if not query_clean:
        return "🤖 **CYBER_AI_ASSISTANT**: Please enter a question or command regarding inventory, forecasts, ROP, or recommendations."

    # -------------------------------------------------------------------
    # 0. GREETINGS & SYSTEM HELP
    # -------------------------------------------------------------------
    if any(re.search(r'\b' + re.escape(k) + r'\b', query_clean) for k in ["hi", "hello", "hey", "help"]) or any(k in query_clean for k in ["who are you", "what can you do"]):
        total_prods = df["product_id"].nunique()
        total_records = len(df)
        return f"""🤖 **// CYBER AI INVENTORY ASSISTANT ONLINE**

I can assist you with real-time supply chain analytics across **{total_prods} products** ({total_records:,} telemetry records).

**Try asking me:**
* 🚨 *"Which products need restock urgently?"*
* 📱 *"Tell me about iPhone 16 Pro"* or *"What is the ROP for Galaxy S24?"*
* 🔥 *"Which product has the highest demand velocity?"*
* 📦 *"Recommend product bundles"*
* 📊 *"Show model accuracy comparison"*
"""

    # -------------------------------------------------------------------
    # 1. RESTOCK & DEFICIT ALERTS
    # -------------------------------------------------------------------
    if any(k in query_clean for k in ["restock", "reorder", "alert", "deficit", "stockout", "out of stock", "urgent"]):
        results = []
        for (brand, pname), pgroup in df.groupby(["brand", "product_name"]):
            pid = pgroup["product_id"].iloc[0]
            lead_time = int(pgroup["lead_time_days"].iloc[0])
            sales = pgroup["units_sold"].tail(14).values
            avg_daily = float(np.mean(sales))
            std_daily = float(np.std(sales)) if float(np.std(sales)) > 0 else 2.0
            
            rop = reorder_point(avg_daily, lead_time, std_daily, 95)
            stock_check = current_stock if (selected_product and pname.lower() == selected_product.lower()) else 60
            needs, qty = restock_decision(stock_check, rop, avg_daily)
            if needs:
                results.append((brand, pname, pid, int(np.ceil(rop)), qty))
                
        if results:
            response = f"🚨 **// CRITICAL RESTOCK ALERTS ({current_stock}-Unit Stock Baseline)**:\n\n"
            for brand, pname, pid, rop_val, qty in results[:10]:
                response += f"• `{pid}` **{brand} {pname}**\n  └ ROP: `{rop_val} units` | Order Qty: **{qty} units**\n\n"
            response += "💡 *Tip: Adjust safety stock slider to boost coverage runway.*"
            return response
        else:
            return "⚡ **// ALL SYSTEMS OPTIMAL**: Current inventory levels exceed Reorder Point thresholds across the entire product catalog. Zero deficits."

    # -------------------------------------------------------------------
    # 2. PRODUCT SPECIFIC INQUIRIES (Exact + Longest Match Scoring)
    # -------------------------------------------------------------------
    all_products = df[["product_id", "brand", "product_name", "category", "unit_price", "lead_time_days"]].drop_duplicates()
    records = all_products.to_dict("records")
    # Sort longest product names first to prevent false partial prefix matches
    records_sorted = sorted(records, key=lambda x: len(x["product_name"]), reverse=True)
    
    matched_prod = None
    
    # Direct match check
    for row in records_sorted:
        pname_lower = row["product_name"].lower()
        pid_lower = row["product_id"].lower()
        brand_lower = row["brand"].lower()
        full_title = f"{brand_lower} {pname_lower}"
        
        if pid_lower == query_clean or pid_lower in query_clean:
            matched_prod = row
            break
        elif pname_lower in query_clean or full_title in query_clean:
            matched_prod = row
            break

    # If no specific product found in query text, check if query refers to active selected product
    if matched_prod is None and selected_product:
        if any(k in query_clean for k in ["this", "current", "selected", "my product", "rop", "safety stock", "eoq", "inventory", "stock", "dossier"]):
            for row in records:
                if row["product_name"].lower() == selected_product.lower():
                    matched_prod = row
                    break

    if matched_prod is not None:
        pid = matched_prod["product_id"]
        brand = matched_prod["brand"]
        pname = matched_prod["product_name"]
        cat = matched_prod["category"]
        price = float(matched_prod["unit_price"])
        if price < 2000:
            price *= 83.0  # USD -> INR conversion
        lead_time = int(matched_prod["lead_time_days"])

        pgroup = df[df["product_id"] == pid].sort_values("date")
        recent_sales = pgroup["units_sold"].tail(30).values
        avg_daily = float(np.mean(recent_sales))
        std_daily = float(np.std(recent_sales)) if float(np.std(recent_sales)) > 0 else 2.0
        
        ss = safety_stock(std_daily, lead_time, 95)
        rop = reorder_point(avg_daily, lead_time, std_daily, 95)
        annual_demand = avg_daily * 365
        eoq = economic_order_quantity(annual_demand, 50.0, price * 0.25)
        dos = days_of_supply(current_stock, avg_daily)

        return f"""📊 **// PRODUCT TELEMETRY DOSSIER: {brand.upper()} {pname.upper()} (`{pid}`)**

* **Category**: {cat}
* **Unit Price**: ₹{price:,.2f}
* **Lead Time**: {lead_time} Days
* **Daily Velocity**: `{avg_daily:.1f} units/day` (30D Mean)
* **Safety Stock (Buffer)**: `{int(np.ceil(ss))} units` (95% Target SLA)
* **Reorder Point (ROP)**: `{int(np.ceil(rop))} units`
* **Economic Order Qty (EOQ)**: `{int(np.ceil(eoq))} units/batch`
* **Days of Supply (at {current_stock} units)**: `{dos:.1f} days runway`
"""

    # -------------------------------------------------------------------
    # 3. TOP DEMAND & VELOCITY LEADERS
    # -------------------------------------------------------------------
    if any(k in query_clean for k in ["top", "highest", "best seller", "demand leader", "velocity", "popular"]):
        top_prods = df.groupby(["product_id", "brand", "product_name"])["units_sold"].mean().reset_index()
        top_prods = top_prods.sort_values("units_sold", ascending=False).head(5)
        
        response = "🔥 **// TOP DEMAND VELOCITY LEADERS (Average Daily Sales)**:\n\n"
        for i, (_, row) in enumerate(top_prods.iterrows(), 1):
            response += f"{i}. **{row['brand']} {row['product_name']}** (`{row['product_id']}`): **{row['units_sold']:.1f} units/day**\n"
        return response

    # -------------------------------------------------------------------
    # 4. PRODUCT BUNDLE RECOMMENDATIONS
    # -------------------------------------------------------------------
    if any(k in query_clean for k in ["bundle", "recommend", "frequently bought", "together", "cross sell"]):
        recs = recommend_bundles(df, top_n=5)
        response = "📦 **// AI RECOMMENDED PRODUCT BUNDLES (Co-Purchase Affinity)**:\n\n"
        for i, r in enumerate(recs, 1):
            sim_pct = int(round(r["similarity"] * 100))
            response += f"{i}. **{r['name_a']}** + **{r['name_b']}** *(Affinity: {sim_pct}%)*\n"
        response += "\n💡 *Strategy: Bundle these complementary items with a 5-10% promo discount to boost basket size.*"
        return response

    # -------------------------------------------------------------------
    # 5. MODEL COMPARISON & ACCURACY
    # -------------------------------------------------------------------
    if any(k in query_clean for k in ["model", "accuracy", "mae", "rmse", "best model", "xgboost", "random forest", "lightgbm", "lstm"]):
        if metrics and "metrics" in metrics:
            best = metrics.get("best_model", "XGBoost")
            response = f"🧠 **// NEURAL ARCHITECTURE SUMMARY**\n\n"
            response += f"🏆 **Top Performing Core**: `{best}`\n\n"
            for name, m in metrics["metrics"].items():
                if isinstance(m, dict) and "MAE" in m:
                    marker = " 👑" if name == best else ""
                    response += f"• **{name}**{marker}\n  └ MAE: `{m['MAE']:.3f}` | RMSE: `{m['RMSE']:.3f}`\n\n"
            return response

    # -------------------------------------------------------------------
    # 6. GENERAL HELP & SYSTEM OVERVIEW
    # -------------------------------------------------------------------
    total_prods = df["product_id"].nunique()
    total_records = len(df)
    
    return f"""🤖 **// CYBER AI INVENTORY ASSISTANT ONLINE**

I can assist you with real-time supply chain analytics across **{total_prods} products** ({total_records:,} telemetry records).

**Try asking me:**
* 🚨 *"Which products need restock urgently?"*
* 📱 *"Tell me about iPhone 16 Pro"* or *"What is the ROP for Galaxy S24?"*
* 🔥 *"Which product has the highest demand velocity?"*
* 📦 *"Recommend product bundles"*
* 📊 *"Show model accuracy comparison"*
"""
