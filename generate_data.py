import numpy as np
import pandas as pd

np.random.seed(42)

dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")

# Catalog: each product has a brand + product name (model), category, base demand,
# lead time, base price, and an optional seasonal multiplier (months -> factor).
products = [
    # ---- Smartphones (5 per brand, recent models) ----
    # Apple
    {"product_id": "P101", "brand": "Apple", "name": "iPhone 15 Pro", "category": "Smartphone", "base_demand": 25, "lead_time_days": 4, "price": 999},
    {"product_id": "P106", "brand": "Apple", "name": "iPhone 16", "category": "Smartphone", "base_demand": 22, "lead_time_days": 4, "price": 1099},
    {"product_id": "P121", "brand": "Apple", "name": "iPhone 16 Pro", "category": "Smartphone", "base_demand": 20, "lead_time_days": 4, "price": 1199},
    {"product_id": "P122", "brand": "Apple", "name": "iPhone 16 Pro Max", "category": "Smartphone", "base_demand": 18, "lead_time_days": 4, "price": 1299},
    {"product_id": "P123", "brand": "Apple", "name": "iPhone 15", "category": "Smartphone", "base_demand": 23, "lead_time_days": 4, "price": 799},
    # Samsung
    {"product_id": "P102", "brand": "Samsung", "name": "Galaxy S24", "category": "Smartphone", "base_demand": 20, "lead_time_days": 3, "price": 799},
    {"product_id": "P107", "brand": "Samsung", "name": "Galaxy S25", "category": "Smartphone", "base_demand": 18, "lead_time_days": 3, "price": 849},
    {"product_id": "P124", "brand": "Samsung", "name": "Galaxy S24 FE", "category": "Smartphone", "base_demand": 14, "lead_time_days": 3, "price": 649},
    {"product_id": "P125", "brand": "Samsung", "name": "Galaxy Z Flip 6", "category": "Smartphone", "base_demand": 10, "lead_time_days": 4, "price": 1099},
    {"product_id": "P126", "brand": "Samsung", "name": "Galaxy Z Fold 6", "category": "Smartphone", "base_demand": 8, "lead_time_days": 4, "price": 1899},
    # Google
    {"product_id": "P108", "brand": "Google", "name": "Pixel 9", "category": "Smartphone", "base_demand": 9, "lead_time_days": 5, "price": 799},
    {"product_id": "P127", "brand": "Google", "name": "Pixel 9 Pro", "category": "Smartphone", "base_demand": 8, "lead_time_days": 5, "price": 999},
    {"product_id": "P128", "brand": "Google", "name": "Pixel 9 Pro XL", "category": "Smartphone", "base_demand": 7, "lead_time_days": 5, "price": 1099},
    {"product_id": "P129", "brand": "Google", "name": "Pixel 9a", "category": "Smartphone", "base_demand": 10, "lead_time_days": 6, "price": 499},
    {"product_id": "P130", "brand": "Google", "name": "Pixel 8a", "category": "Smartphone", "base_demand": 9, "lead_time_days": 6, "price": 499},
    # Xiaomi
    {"product_id": "P109", "brand": "Xiaomi", "name": "Redmi Note 14 Pro", "category": "Smartphone", "base_demand": 15, "lead_time_days": 6, "price": 329},
    {"product_id": "P131", "brand": "Xiaomi", "name": "Xiaomi 14", "category": "Smartphone", "base_demand": 11, "lead_time_days": 6, "price": 749},
    {"product_id": "P132", "brand": "Xiaomi", "name": "Xiaomi 14 Ultra", "category": "Smartphone", "base_demand": 9, "lead_time_days": 6, "price": 1099},
    {"product_id": "P133", "brand": "Xiaomi", "name": "Redmi Note 13", "category": "Smartphone", "base_demand": 14, "lead_time_days": 6, "price": 249},
    {"product_id": "P134", "brand": "Xiaomi", "name": "POCO X6 Pro", "category": "Smartphone", "base_demand": 12, "lead_time_days": 6, "price": 349},
    # OnePlus
    {"product_id": "P110", "brand": "OnePlus", "name": "OnePlus 13", "category": "Smartphone", "base_demand": 10, "lead_time_days": 5, "price": 699},
    {"product_id": "P135", "brand": "OnePlus", "name": "OnePlus 12", "category": "Smartphone", "base_demand": 10, "lead_time_days": 5, "price": 599},
    {"product_id": "P136", "brand": "OnePlus", "name": "OnePlus 12R", "category": "Smartphone", "base_demand": 9, "lead_time_days": 5, "price": 499},
    {"product_id": "P137", "brand": "OnePlus", "name": "OnePlus Nord 4", "category": "Smartphone", "base_demand": 12, "lead_time_days": 5, "price": 449},
    {"product_id": "P138", "brand": "OnePlus", "name": "OnePlus Nord CE 4", "category": "Smartphone", "base_demand": 11, "lead_time_days": 5, "price": 329},
    # Motorola
    {"product_id": "P111", "brand": "Motorola", "name": "Edge 50 Pro", "category": "Smartphone", "base_demand": 8, "lead_time_days": 7, "price": 599},
    {"product_id": "P139", "brand": "Motorola", "name": "Moto G85", "category": "Smartphone", "base_demand": 12, "lead_time_days": 7, "price": 299},
    {"product_id": "P140", "brand": "Motorola", "name": "Edge 50 Fusion", "category": "Smartphone", "base_demand": 9, "lead_time_days": 7, "price": 399},
    {"product_id": "P141", "brand": "Motorola", "name": "Moto Razr 50", "category": "Smartphone", "base_demand": 7, "lead_time_days": 7, "price": 899},
    {"product_id": "P142", "brand": "Motorola", "name": "Moto G54", "category": "Smartphone", "base_demand": 11, "lead_time_days": 7, "price": 249},

    # ---- Refrigerators (Appliance) ----
    {"product_id": "P103", "brand": "Samsung", "name": "Smart Refrigerator 450L", "category": "Refrigerator", "base_demand": 8, "lead_time_days": 7, "price": 1250},
    {"product_id": "P112", "brand": "LG", "name": "InstaView Refrigerator 530L", "category": "Refrigerator", "base_demand": 7, "lead_time_days": 7, "price": 1450},
    {"product_id": "P113", "brand": "Whirlpool", "name": "Double Door Refrigerator 400L", "category": "Refrigerator", "base_demand": 9, "lead_time_days": 8, "price": 880},

    # ---- Air Conditioners (Appliance, summer seasonal) ----
    {"product_id": "P104", "brand": "LG", "name": "1.5T Inverter AC", "category": "Air Conditioner", "base_demand": 12, "lead_time_days": 6, "price": 620, "seasonal_months": [4, 5, 6, 7], "seasonal_factor": 2.0},
    {"product_id": "P114", "brand": "Samsung", "name": "2T Inverter AC", "category": "Air Conditioner", "base_demand": 9, "lead_time_days": 6, "price": 780, "seasonal_months": [4, 5, 6, 7], "seasonal_factor": 2.0},

    # ---- Televisions (Appliance) ----
    {"product_id": "P105", "brand": "LG", "name": "4K Smart OLED TV 55\"", "category": "Television", "base_demand": 10, "lead_time_days": 5, "price": 1100},
    {"product_id": "P115", "brand": "Sony", "name": "Bravia 4K Smart TV 65\"", "category": "Television", "base_demand": 8, "lead_time_days": 6, "price": 1400},

    # ---- Washing Machines (Appliance) ----
    {"product_id": "P116", "brand": "Samsung", "name": "Front-Load Washer 8kg", "category": "Washing Machine", "base_demand": 9, "lead_time_days": 7, "price": 720},
    {"product_id": "P117", "brand": "LG", "name": "Top-Load Washer 7kg", "category": "Washing Machine", "base_demand": 8, "lead_time_days": 6, "price": 480},

    # ---- Kitchen / Small Appliances ----
    {"product_id": "P118", "brand": "LG", "name": "Microwave Oven 32L", "category": "Kitchen Appliance", "base_demand": 11, "lead_time_days": 4, "price": 220},
    {"product_id": "P119", "brand": "Samsung", "name": "Air Purifier", "category": "Air Purifier", "base_demand": 7, "lead_time_days": 5, "price": 350, "seasonal_months": [11, 12, 1, 2], "seasonal_factor": 1.5},
    {"product_id": "P120", "brand": "Bosch", "name": "Freestanding Dishwasher", "category": "Dishwasher", "base_demand": 6, "lead_time_days": 8, "price": 950},
]

rows = []

for prod in products:
    for date in dates:
        demand = prod["base_demand"]

        if date.weekday() in [5, 6]:
            demand *= 1.35

        # Product-specific seasonal months (e.g. ACs in summer, purifiers in winter)
        if "seasonal_months" in prod and date.month in prod["seasonal_months"]:
            demand *= prod["seasonal_factor"]

        if date.month in [11, 12]:
            demand *= 1.4

        on_promo = 1 if np.random.rand() < 0.10 else 0
        if on_promo:
            demand *= 1.5

        units_sold = max(0, int(np.random.normal(loc=demand, scale=demand * 0.05)))

        price = prod["price"] * 83.0  # Convert USD to INR (Rupees)
        if on_promo:
            price *= 0.85

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "product_id": prod["product_id"],
            "brand": prod["brand"],
            "product_name": prod["name"],
            "category": prod["category"],
            "lead_time_days": prod["lead_time_days"],
            "on_promo": on_promo,
            "unit_price": round(price, 2),
            "units_sold": units_sold
        })

df = pd.DataFrame(rows)
df.to_csv("retail_sales_data.csv", index=False)
print(f"[+] Success! 'retail_sales_data.csv' created with {len(df)} records across {df['product_id'].nunique()} products.")
