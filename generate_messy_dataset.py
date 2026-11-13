#!/usr/bin/env python3
"""
Generate a Massive, Multi-Layered Messy Enterprise Dataset for testing Agentic Preprocessing.
Produces a rich CSV with 20,000+ rows and 22 columns spanning:
- Schema anomalies & mixed types
- Currency string parsing & sentinel tokens
- Multiple conflicting date formats
- Temporal precedence inversions (ship_date < order_date)
- Mathematical invariant breaches (price * qty != subtotal, subtotal + tax != total)
- Fuzzy near-duplicate cities & countries
- True Primary Key collisions (same ID, conflicting records)
- Exact duplicate rows
- Extreme outliers & range violations
- Heterogeneous boolean representations
"""

import os
import random
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def create_massive_messy_dataset(num_rows: int = 20000, output_path: str = "test_data/massive_messy_dataset.csv"):
    random.seed(42)
    np.random.seed(42)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Generating massive messy dataset with ~{num_rows:,} base rows...")

    # Reference pools
    first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
                   "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
                   "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
                   "Matthew", "Betty", "Anthony", "Margaret", "Donald", "Sandra", "München", "François"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
                  "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"]

    city_variants = {
        "San Francisco": ["San Francisco", "San Fransisco", "San Francisco, CA", "SF", "  San Francisco  ", "san francisco"],
        "New York": ["New York", "New York City", "New York, NY", "NYC", "Nw York", "new york", "New York "],
        "Los Angeles": ["Los Angeles", "Los Angelos", "LA", "Los Angeles, CA", "los angeles", " Los Angeles"],
        "Chicago": ["Chicago", "Chicagoo", "Chicago, IL", "chicago", "  Chicago\t"],
        "Austin": ["Austin", "Austin, TX", "Austn", "austin", "Austin "],
        "Seattle": ["Seattle", "Seatle", "Seattle, WA", "seattle", "\tSeattle"],
        "Miami": ["Miami", "Miami, FL", "Miamii", "miami"],
        "London": ["London", "London, UK", "Greater London", "london"]
    }

    country_variants = {
        "USA": ["USA", "US", "United States", "United States of America", "U.S.", "usa", "  United States  "],
        "UK": ["UK", "United Kingdom", "Great Britain", "GB", "uk", "England"],
        "Canada": ["Canada", "CA", "CAN", "canada", " Canada "],
        "Germany": ["Germany", "DE", "Deutschland", "germany", "DEU"]
    }

    categories = [
        "Electronics", "Apparel", "Home & Kitchen", "Books",
        "Sports & Outdoors", "Health & Personal Care", "Automotive", "Toys & Games"
    ]

    skus = [f"SKU-{1000 + i}" for i in range(150)]

    payment_variants = [
        "Credit Card", "credit_card", "CREDIT CARD", "CC", "Visa", "MasterCard",
        "PayPal", "paypal", "PAYPAL", "Apple Pay", "apple_pay", "Google Pay", "Wire Transfer", "N/A"
    ]

    boolean_variants = [
        "True", "False", "true", "false", "TRUE", "FALSE",
        "1", "0", 1, 0, True, False, "yes", "no", "YES", "NO", "Y", "N", "t", "f", ""
    ]

    sentinels = ["N/A", "null", "NULL", "None", "-999", "UNKNOWN", "missing", "nan", ""]

    base_start_date = datetime(2023, 1, 1)

    records = []

    for i in range(num_rows):
        txn_id = f"TXN-{100000 + i}"
        cust_id = f"CUST-{random.randint(1000, 4500)}"
        first = random.choice(first_names)
        last = random.choice(last_names)
        
        # 1. Customer Name with messy whitespace & casing
        if random.random() < 0.25:
            ws_type = random.choice(["  ", "\t", "   "])
            cust_name = f"{ws_type}{first} {last}{ws_type}"
        elif random.random() < 0.10:
            cust_name = f"{first} {last}".upper() if random.random() < 0.5 else f"{first} {last}".lower()
        elif random.random() < 0.05:
            cust_name = random.choice(sentinels)
        else:
            cust_name = f"{first} {last}"

        # 2. Customer Email (with broken formats & nulls)
        if random.random() < 0.08:
            cust_email = None
        elif random.random() < 0.04:
            cust_email = f"{first.lower()}.{last.lower()}gmail.com"  # missing @
        elif random.random() < 0.03:
            cust_email = random.choice(sentinels)
        else:
            cust_email = f"{first.lower()}_{last.lower()}@{random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'company.org'])}"

        # 3. City & Country with fuzzy near-duplicates
        city_key = random.choice(list(city_variants.keys()))
        city = random.choice(city_variants[city_key])
        country_key = "UK" if city_key == "London" else "USA"
        country = random.choice(country_variants[country_key])

        # 4. Dates & Mixed Formats
        days_offset = random.randint(0, 650)
        dt_order = base_start_date + timedelta(days=days_offset, hours=random.randint(8, 20), minutes=random.randint(0, 59))
        
        # Order date representation
        fmt_choice = random.random()
        if fmt_choice < 0.50:
            order_date_str = dt_order.strftime("%Y-%m-%d")
        elif fmt_choice < 0.70:
            order_date_str = dt_order.strftime("%m/%d/%Y")
        elif fmt_choice < 0.85:
            order_date_str = dt_order.strftime("%d-%b-%Y")
        elif fmt_choice < 0.95:
            order_date_str = dt_order.strftime("%Y/%m/%d %H:%M:%S")
        else:
            order_date_str = str(int(dt_order.timestamp()))  # epoch timestamp

        # Ship Date (intentional temporal inversion for ~7% of rows)
        if random.random() < 0.07:
            # Anomaly: Ship date PRECEDES order date
            dt_ship = dt_order - timedelta(days=random.randint(1, 5))
        else:
            dt_ship = dt_order + timedelta(days=random.randint(1, 4))
        
        if random.random() < 0.06:
            ship_date_str = None
        else:
            ship_date_str = dt_ship.strftime("%Y-%m-%d" if random.random() < 0.7 else "%m/%d/%Y")

        # Delivery Date
        if ship_date_str is None or random.random() < 0.12:
            delivery_date_str = None
        elif random.random() < 0.05:
            # Anomaly: Delivery before ship
            dt_deliv = dt_ship - timedelta(days=random.randint(1, 3))
            delivery_date_str = dt_deliv.strftime("%Y-%m-%d")
        else:
            dt_deliv = dt_ship + timedelta(days=random.randint(1, 7))
            delivery_date_str = dt_deliv.strftime("%Y-%m-%d")

        # 5. Product & Category
        sku = random.choice(skus) if random.random() > 0.03 else random.choice(sentinels)
        cat = random.choice(categories)
        if random.random() < 0.15:
            cat = f"  {cat}  "
        elif random.random() < 0.08:
            cat = cat.lower()

        # 6. Pricing, Quantities & Currency formatting
        base_unit_price = round(random.uniform(5.0, 750.0), 2)
        qty = random.randint(1, 8)

        # Quantity anomalies
        if random.random() < 0.03:
            qty_val = -random.randint(1, 3)  # Negative quantity!
        elif random.random() < 0.05:
            qty_val = f"{qty}.0"
        elif random.random() < 0.03:
            qty_val = f"{qty} units"
        elif random.random() < 0.02:
            qty_val = random.choice(sentinels)
        else:
            qty_val = qty

        # Price formatting anomalies
        price_mode = random.random()
        if price_mode < 0.35:
            unit_price_val = f"${base_unit_price:,.2f}"
        elif price_mode < 0.50:
            unit_price_val = f"$ {base_unit_price:.2f}"
        elif price_mode < 0.58:
            unit_price_val = f"USD {base_unit_price:.2f}"
        elif price_mode < 0.63:
            unit_price_val = f"-${base_unit_price:.2f}"  # Negative price!
        elif price_mode < 0.70:
            unit_price_val = random.choice(sentinels)
        elif price_mode < 0.76:
            unit_price_val = None
        else:
            unit_price_val = base_unit_price

        # 7. Discount Percentages
        disc_num = random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
        if random.random() < 0.02:
            disc_val = "150%"  # Outlier
        elif random.random() < 0.02:
            disc_val = "-10%"  # Negative percentage
        elif random.random() < 0.40:
            disc_val = f"{int(disc_num * 100)}%"
        elif random.random() < 0.55:
            disc_val = f"{disc_num * 100:.1f}%"
        elif random.random() < 0.65:
            disc_val = random.choice(sentinels)
        elif random.random() < 0.75:
            disc_val = None
        else:
            disc_val = disc_num

        # 8. Arithmetic calculation & intentional violations
        calc_qty = qty if isinstance(qty_val, int) and qty_val > 0 else 1
        true_subtotal = round(base_unit_price * calc_qty * (1 - disc_num), 2)

        if random.random() < 0.09:
            # Anomaly: Math mismatch (unit_price * qty != subtotal)
            line_subtotal_val = round(true_subtotal * random.uniform(1.2, 1.8), 2)
        elif random.random() < 0.25:
            line_subtotal_val = f"${true_subtotal:,.2f}"
        elif random.random() < 0.05:
            line_subtotal_val = random.choice(sentinels)
        else:
            line_subtotal_val = true_subtotal

        # Tax amount & Grand Total
        tax_val = round(true_subtotal * 0.0825, 2) if random.random() > 0.08 else None
        calc_tax = tax_val if isinstance(tax_val, (int, float)) else 0.0
        true_grand_total = round(true_subtotal + calc_tax, 2)

        if random.random() < 0.08:
            # Anomaly: Math mismatch (subtotal + tax != grand_total)
            grand_total_val = round(true_grand_total + random.uniform(25.0, 150.0), 2)
        elif random.random() < 0.02:
            grand_total_val = 9999999.00  # Extreme Outlier!
        elif random.random() < 0.35:
            grand_total_val = f"${true_grand_total:,.2f}"
        elif random.random() < 0.05:
            grand_total_val = random.choice(sentinels)
        else:
            grand_total_val = true_grand_total

        # 9. Payment & Expedited shipping
        pay_method = random.choice(payment_variants)
        expedited = random.choice(boolean_variants)

        # 10. Customer Ratings (1.0 to 5.0 with outliers)
        if random.random() < 0.20:
            rating_val = None
        elif random.random() < 0.02:
            rating_val = 99.0  # Impossible rating!
        elif random.random() < 0.01:
            rating_val = -1.0  # Negative rating
        else:
            rating_val = round(random.choice([1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0]), 1)

        # 11. Delivery Status & Notes
        status_opts = ["Delivered", "In Transit", "Shipped", "Cancelled", "Returned", "Pending"]
        status = random.choice(status_opts)
        if random.random() < 0.12:
            status = f"  {status.lower()}  "
        elif random.random() < 0.04:
            status = random.choice(sentinels)

        note_snippets = [
            "Customer requested doorstep dropoff.",
            "Signature verified by recipient.",
            "Expedited warehouse pick.",
            "Promo code applied at checkout: SUMMER2024",
            "Billing address matches shipping destination.",
            "Flagged for audit review: high item volume.",
            "Special handling: fragile glass contents.",
            "Gift wrap requested; no receipt included in package.",
            "Recurring monthly subscription order.",
            "Payment re-authorized after temporary gateway timeout."
        ]
        notes = random.choice(note_snippets) if random.random() > 0.25 else None

        records.append({
            "transaction_id": txn_id,
            "order_date": order_date_str,
            "ship_date": ship_date_str,
            "delivery_date": delivery_date_str,
            "customer_id": cust_id,
            "customer_name": cust_name,
            "customer_email": cust_email,
            "customer_city": city,
            "customer_country": country,
            "product_sku": sku,
            "product_category": cat,
            "unit_price": unit_price_val,
            "quantity": qty_val,
            "discount_pct": disc_val,
            "line_subtotal": line_subtotal_val,
            "tax_amount": tax_val,
            "grand_total": grand_total_val,
            "payment_method": pay_method,
            "is_expedited_shipping": expedited,
            "customer_rating": rating_val,
            "delivery_status": status,
            "internal_notes": notes
        })

    df = pd.DataFrame(records)

    # ── 12. Inject 350 Exact Duplicate Rows ──
    print("Injecting 350 exact duplicate rows...")
    dup_indices = np.random.choice(len(df), size=350, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # ── 13. Inject 120 Primary Key Collisions ──
    print("Injecting 120 Primary Key collisions (same transaction_id, conflicting data)...")
    collision_indices = np.random.choice(len(df) - 350, size=120, replace=False)
    for idx in collision_indices:
        colliding_row = df.iloc[idx].to_dict()
        # Same transaction_id, but completely conflicting customer and amount
        colliding_row["customer_id"] = f"CUST-CONFLICT-{random.randint(8000, 9999)}"
        colliding_row["customer_name"] = "CONFLICTING ENTITY LTD"
        colliding_row["customer_city"] = "Tokyo"
        colliding_row["grand_total"] = "$999.99"
        colliding_row["internal_notes"] = "PRIMARY KEY COLLISION INJECTED FOR AUDIT TEST"
        df = pd.concat([df, pd.DataFrame([colliding_row])], ignore_index=True)

    # Shuffle the dataset thoroughly
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("=" * 60)
    print(f"✓ Massive Messy Dataset Created Successfully!")
    print(f"📍 Location:     {output_path}")
    print(f"📊 Total Rows:   {len(df):,}")
    print(f"📋 Total Columns:{len(df.columns)}")
    print(f"💾 File Size:    {file_size_mb:.2f} MB")
    print("=" * 60)
    print("Included Anomaly Layers:")
    print(" • Currency strings & symbols ('$1,249.99', 'USD 50', '-$25')")
    print(" • 7 Sentinel tokens ('N/A', '-999', 'null', 'UNKNOWN', etc.)")
    print(" • 5 Heterogeneous date formats & epoch timestamps")
    print(" • Inverted temporal order (ship_date < order_date)")
    print(" • Arithmetic violations (unit_price * qty != subtotal, subtotal + tax != grand_total)")
    print(" • 8 Fuzzy near-duplicate city clusters ('San Francisco', 'SF', 'San Fransisco')")
    print(" • 120 Primary Key collisions with contradictory payload")
    print(" • 350 Exact duplicate multi-column rows")
    print(" • Extreme outliers ($9,999,999.00 total, 99.0 customer rating, 150% discount)")
    print(" • Heterogeneous boolean flags ('True', '1', 'yes', 'Y', 'false', '0', 'no')")
    print("=" * 60)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a massive messy dataset for AI preprocessing.")
    parser.add_argument("--rows", type=int, default=20000, help="Number of base rows to generate (default: 20000)")
    parser.add_argument("--output", type=str, default="test_data/massive_messy_dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    create_massive_messy_dataset(num_rows=args.rows, output_path=args.output)

