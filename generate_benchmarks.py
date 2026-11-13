"""
Synthetic Benchmark Generator for Agentic Data Preprocessing System.
Generates 6 production-scale messy datasets (~5,000 to 10,000 rows each)
with calibrated data quality anomalies across 6 different industry verticals.
"""

import os
import random
import datetime
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

OUT_DIR = "benchmark_data"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. E-Commerce Orders (10,000 rows) ────────────────────────────────────────
def generate_ecommerce_orders(n=10000):
    print(f"Generating {n:,} rows for E-Commerce Orders...")
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
    fuzzy_map = {
        "New York": ["New York City", "NYC", "New-York"],
        "Chicago": ["Chicagoo", "Chicago IL"],
        "San Francisco": ["San Fransisco", "SF", "San Fran"],
        "Los Angeles": ["L.A.", "Los Angeles CA"]
    }
    
    order_ids = [100000 + i for i in range(n)]
    customers = [f"CUST_{random.randint(1000, 5000)}" for _ in range(n)]
    
    base_date = datetime.datetime(2026, 1, 1)
    order_dates = [base_date + datetime.timedelta(days=random.randint(0, 180), minutes=random.randint(0, 1440)) for _ in range(n)]
    # Normally ship is 1-5 days after order
    ship_dates = [od + datetime.timedelta(days=random.randint(1, 5), hours=random.randint(1, 12)) for od in order_dates]
    
    subtotals = [round(random.uniform(15.0, 500.0), 2) for _ in range(n)]
    taxes = [round(s * 0.08, 2) for s in subtotals]
    totals = [round(s + t, 2) for s, t in zip(subtotals, taxes)]
    
    emails = [f"user_{i}@example.com" for i in range(n)]
    city_col = [random.choice(cities) for _ in range(n)]
    
    df = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customers,
        "customer_email": emails,
        "customer_city": city_col,
        "order_date": [d.strftime("%Y-%m-%d %H:%M:%S") for d in order_dates],
        "ship_date": [d.strftime("%Y-%m-%d %H:%M:%S") for d in ship_dates],
        "subtotal": subtotals,
        "tax_amount": taxes,
        "total_amount": totals,
    })
    
    # ── Inject Anomalies ──
    # 1. Exact duplicates (~300 rows)
    dupe_indices = random.sample(range(n), 300)
    dupe_rows = df.iloc[dupe_indices].copy()
    df = pd.concat([df, dupe_rows], ignore_index=True)
    
    # 2. Key conflicts on order_id (~100 rows): same order_id, different amount/customer
    conflict_indices = random.sample(range(1000, 5000), 100)
    for idx in conflict_indices:
        new_row = df.iloc[idx].copy()
        new_row["total_amount"] = round(new_row["total_amount"] * 1.5, 2)
        new_row["customer_id"] = "CUST_CONFLICT"
        df.loc[len(df)] = new_row
        
    # 3. Temporal inversions (~150 rows): ship_date < order_date
    temp_inv = random.sample(range(n), 150)
    for idx in temp_inv:
        od = datetime.datetime.strptime(df.at[idx, "order_date"], "%Y-%m-%d %H:%M:%S")
        df.at[idx, "ship_date"] = (od - datetime.timedelta(days=random.randint(1, 4))).strftime("%Y-%m-%d %H:%M:%S")
        
    # 4. Arithmetic violations (~120 rows): total != subtotal + tax
    math_inv = random.sample(range(n), 120)
    for idx in math_inv:
        df.at[idx, "total_amount"] = round(df.at[idx, "total_amount"] + random.uniform(10.0, 50.0), 2)
        
    # 5. Fuzzy city names (~250 rows)
    for city, variants in fuzzy_map.items():
        match_idx = df[df["customer_city"] == city].index
        if len(match_idx) > 50:
            chosen = random.sample(list(match_idx), 50)
            for c in chosen:
                df.at[c, "customer_city"] = random.choice(variants)
                
    # 6. Invalid email format (~80 rows)
    bad_emails = random.sample(range(n), 80)
    for idx in bad_emails:
        df.at[idx, "customer_email"] = random.choice(["invalid_email", "user@domain", "plainaddress", "test@.com"])
        
    # 7. Null order totals (~100 rows)
    null_totals = random.sample(range(n), 100)
    for idx in null_totals:
        df.at[idx, "total_amount"] = None

    path = os.path.join(OUT_DIR, "ecommerce_orders.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df):,} rows)")
    return path


# ── 2. Healthcare Patients (5,000 rows) ───────────────────────────────────────
def generate_healthcare_patients(n=5000):
    print(f"Generating {n:,} rows for Healthcare Patients...")
    blood_types = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    departments = ["Cardiology", "Neurology", "Pediatrics", "Oncology", "Orthopedics", "Emergency"]
    fuzzy_depts = {
        "Cardiology": ["Cardio", "Cardiology Dept"],
        "Emergency": ["Emergency Room", "ER", "Emerg Dept"],
        "Pediatrics": ["Pediatric", "Pediatrics Unit"]
    }
    
    patient_ids = [200000 + i for i in range(n)]
    ages = [random.randint(1, 85) for _ in range(n)]
    bmis = [round(random.uniform(18.5, 34.0), 1) for _ in range(n)]
    sys_bps = [random.randint(110, 160) for _ in range(n)]
    dia_bps = [int(s * 0.65) for s in sys_bps]
    
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "patient_age": ages,
        "bmi": bmis,
        "systolic_bp": sys_bps,
        "diastolic_bp": dia_bps,
        "blood_type": [random.choice(blood_types) for _ in range(n)],
        "admission_department": [random.choice(departments) for _ in range(n)],
        "admission_date": [(datetime.datetime(2026, 1, 1) + datetime.timedelta(days=random.randint(0, 120))).strftime("%Y-%m-%d") for _ in range(n)],
        "discharge_date": [(datetime.datetime(2026, 1, 1) + datetime.timedelta(days=random.randint(121, 150))).strftime("%Y-%m-%d") for _ in range(n)],
    })
    
    # ── Inject Anomalies ──
    # 1. Negative or impossible ages (~60 rows)
    bad_ages = random.sample(range(n), 60)
    for idx in bad_ages[:30]:
        df.at[idx, "patient_age"] = -random.randint(1, 15)
    for idx in bad_ages[30:]:
        df.at[idx, "patient_age"] = random.randint(130, 250)
        
    # 2. Impossible BMIs (~50 rows)
    bad_bmi = random.sample(range(n), 50)
    for idx in bad_bmi:
        df.at[idx, "bmi"] = -random.uniform(5.0, 20.0) if random.random() < 0.5 else random.uniform(180.0, 350.0)
        
    # 3. Missing primary keys (~40 rows)
    null_pks = random.sample(range(n), 40)
    for idx in null_pks:
        df.at[idx, "patient_id"] = None
        
    # 4. Fuzzy department names (~150 rows)
    for dept, variants in fuzzy_depts.items():
        match_idx = df[df["admission_department"] == dept].index
        if len(match_idx) > 40:
            for c in random.sample(list(match_idx), 40):
                df.at[c, "admission_department"] = random.choice(variants)
                
    # 5. Exact duplicates (~150 rows)
    dupes = df.iloc[random.sample(range(n), 150)].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    path = os.path.join(OUT_DIR, "healthcare_patients.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df):,} rows)")
    return path


# ── 3. Fintech Transactions (10,000 rows) ────────────────────────────────────
def generate_fintech_transactions(n=10000):
    print(f"Generating {n:,} rows for Fintech Transactions...")
    txn_ids = [f"TXN_{300000 + i}" for i in range(n)]
    accounts = [f"ACC_{random.randint(10000, 99999)}" for _ in range(n)]
    tx_types = ["TRANSFER", "PAYMENT", "WITHDRAWAL", "DEPOSIT"]
    
    amounts = [round(random.uniform(5.0, 5000.0), 2) for _ in range(n)]
    fees = [round(a * 0.015, 2) for a in amounts]
    total_debited = [round(a + f, 2) for a, f in zip(amounts, fees)]
    
    df = pd.DataFrame({
        "transaction_id": txn_ids,
        "account_id": accounts,
        "transaction_type": [random.choice(tx_types) for _ in range(n)],
        "amount": amounts,
        "fee": fees,
        "total_debited": total_debited,
        "created_at": [(datetime.datetime(2026, 1, 1) + datetime.timedelta(seconds=random.randint(0, 15000000))).strftime("%Y-%m-%d %H:%M:%S") for _ in range(n)]
    })
    
    # ── Inject Anomalies ──
    # 1. Math discrepancy: amount + fee != total_debited (~180 rows)
    bad_math = random.sample(range(n), 180)
    for idx in bad_math:
        df.at[idx, "total_debited"] = round(df.at[idx, "total_debited"] + random.uniform(5.0, 100.0), 2)
        
    # 2. Sentinels in amounts (~90 rows: string "N/A", "-999")
    df["amount"] = df["amount"].astype(object)
    bad_sentinels = random.sample(range(n), 90)
    for idx in bad_sentinels:
        df.at[idx, "amount"] = random.choice(["N/A", "-999", "NULL", "unknown"])
        
    # 3. Untrimmed whitespace in account_id (~300 rows)
    ws_idx = random.sample(range(n), 300)
    for idx in ws_idx:
        df.at[idx, "account_id"] = f"  {df.at[idx, 'account_id']}  "
        
    # 4. Key collisions on transaction_id (~120 rows)
    coll_idx = random.sample(range(n), 120)
    for idx in coll_idx:
        row = df.iloc[idx].copy()
        row["account_id"] = "ACC_CONFLICT"
        row["fee"] = 99.99
        df.loc[len(df)] = row

    path = os.path.join(OUT_DIR, "fintech_transactions.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df):,} rows)")
    return path


# ── 4. Bikeshare Trips (10,000 rows) ─────────────────────────────────────────
def generate_bikeshare_trips(n=10000):
    print(f"Generating {n:,} rows for Bikeshare Trips...")
    ride_ids = [f"RIDE_{400000 + i}" for i in range(n)]
    rideable_types = ["classic_bike", "electric_bike", "dockless_scooter"]
    
    stations = [f"Station {i}" for i in range(1, 100)]
    
    base_time = datetime.datetime(2026, 3, 1)
    starts = [base_time + datetime.timedelta(minutes=random.randint(0, 40000)) for _ in range(n)]
    durations = [random.randint(180, 7200) for _ in range(n)]
    ends = [s + datetime.timedelta(seconds=dur) for s, dur in zip(starts, durations)]
    
    start_lats = [round(random.uniform(41.80, 42.00), 5) for _ in range(n)]
    start_lngs = [round(random.uniform(-87.75, -87.55), 5) for _ in range(n)]
    end_lats = [round(lat + random.uniform(-0.02, 0.02), 5) for lat in start_lats]
    end_lngs = [round(lng + random.uniform(-0.02, 0.02), 5) for lng in start_lngs]
    
    df = pd.DataFrame({
        "ride_id": ride_ids,
        "rideable_type": [random.choice(rideable_types) for _ in range(n)],
        "started_at": [s.strftime("%Y-%m-%d %H:%M:%S") for s in starts],
        "ended_at": [e.strftime("%Y-%m-%d %H:%M:%S") for e in ends],
        "start_station_name": [random.choice(stations) for _ in range(n)],
        "end_station_name": [random.choice(stations) for _ in range(n)],
        "start_lat": start_lats,
        "start_lng": start_lngs,
        "end_lat": end_lats,
        "end_lng": end_lngs,
        "duration_seconds": durations
    })
    
    # ── Inject Anomalies ──
    # 1. Inverted trip timestamps (~150 rows: ended_at < started_at)
    inv_idx = random.sample(range(n), 150)
    for idx in inv_idx:
        st = datetime.datetime.strptime(df.at[idx, "started_at"], "%Y-%m-%d %H:%M:%S")
        df.at[idx, "ended_at"] = (st - datetime.timedelta(minutes=random.randint(5, 30))).strftime("%Y-%m-%d %H:%M:%S")
        
    # 2. Out of bounds Lat/Lng (~80 rows: lat > 90, lng < -180)
    bad_gps = random.sample(range(n), 80)
    for idx in bad_gps[:40]:
        df.at[idx, "start_lat"] = random.choice([142.5, 999.0, -95.0])
    for idx in bad_gps[40:]:
        df.at[idx, "end_lng"] = random.choice([-220.0, 340.0])
        
    # 3. Structural missingness: dockless scooters have null stations (~1200 rows)
    scooter_mask = df["rideable_type"] == "dockless_scooter"
    df.loc[scooter_mask, "start_station_name"] = None
    df.loc[scooter_mask, "end_station_name"] = None
    
    # 4. Exact duplicates (~200 rows)
    dupes = df.iloc[random.sample(range(n), 200)].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    path = os.path.join(OUT_DIR, "bikeshare_trips.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df):,} rows)")
    return path


# ── 5. HR Employee Attrition (5,000 rows) ────────────────────────────────────
def generate_hr_employees(n=5000):
    print(f"Generating {n:,} rows for HR Employee Attrition...")
    depts = ["Sales", "Engineering", "Marketing", "Human Resources", "Finance", "Product"]
    fuzzy_depts = {
        "Human Resources": ["HR", "H.R.", "People Operations"],
        "Sales": ["Sales Dept", "sales", "Direct Sales"],
        "Engineering": ["Eng", "Software Engineering"]
    }
    
    emp_ids = [500000 + i for i in range(n)]
    salaries = [random.randint(45000, 220000) for _ in range(n)]
    
    base_hire = datetime.datetime(2018, 1, 1)
    hire_dates = [base_hire + datetime.timedelta(days=random.randint(0, 2000)) for _ in range(n)]
    exit_dates = []
    statuses = []
    for h in hire_dates:
        if random.random() < 0.25:
            exit_dates.append((h + datetime.timedelta(days=random.randint(60, 1000))).strftime("%Y-%m-%d"))
            statuses.append("Terminated")
        else:
            exit_dates.append(None)
            statuses.append("Active")
            
    df = pd.DataFrame({
        "employee_id": emp_ids,
        "department": [random.choice(depts) for _ in range(n)],
        "annual_salary": salaries,
        "employment_status": statuses,
        "hire_date": [h.strftime("%Y-%m-%d") for h in hire_dates],
        "exit_date": exit_dates,
        "performance_rating": [random.randint(1, 5) for _ in range(n)]
    })
    
    # ── Inject Anomalies ──
    # 1. Negative salaries (~40 rows)
    bad_sal = random.sample(range(n), 40)
    for idx in bad_sal:
        df.at[idx, "annual_salary"] = -abs(df.at[idx, "annual_salary"])
        
    # 2. Inverted dates: exit_date < hire_date (~35 rows)
    term_idx = df[df["employment_status"] == "Terminated"].index
    if len(term_idx) > 35:
        inv_chosen = random.sample(list(term_idx), 35)
        for idx in inv_chosen:
            h = datetime.datetime.strptime(df.at[idx, "hire_date"], "%Y-%m-%d")
            df.at[idx, "exit_date"] = (h - datetime.timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")
            
    # 3. Fuzzy department names (~200 rows)
    for dept, variants in fuzzy_depts.items():
        match_idx = df[df["department"] == dept].index
        if len(match_idx) > 50:
            for c in random.sample(list(match_idx), 50):
                df.at[c, "department"] = random.choice(variants)
                
    # 4. Out of range performance ratings (~25 rows: rating > 5 or < 1)
    bad_ratings = random.sample(range(n), 25)
    for idx in bad_ratings:
        df.at[idx, "performance_rating"] = random.choice([0, 8, 9, 10])

    path = os.path.join(OUT_DIR, "hr_employee_attrition.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df):,} rows)")
    return path


# ── 6. SaaS Subscription Churn (8,000 rows) ──────────────────────────────────
def generate_saas_subscriptions(n=8000):
    print(f"Generating {n:,} rows for SaaS Subscription Churn...")
    sub_ids = [f"SUB_{600000 + i}" for i in range(n)]
    plans = ["Basic", "Professional", "Enterprise"]
    plan_fees = {"Basic": 29.0, "Professional": 99.0, "Enterprise": 299.0}
    
    tenures = [random.randint(1, 36) for _ in range(n)]
    chosen_plans = [random.choice(plans) for _ in range(n)]
    monthly_charges = [plan_fees[p] for p in chosen_plans]
    total_charges = [round(m * t, 2) for m, t in zip(monthly_charges, tenures)]
    
    df = pd.DataFrame({
        "subscription_id": sub_ids,
        "plan_tier": chosen_plans,
        "tenure_months": tenures,
        "monthly_charge": monthly_charges,
        "total_charges": total_charges,
        "cancellation_reason": [random.choice(["Too expensive", "Missing features", "Switched competitor", None, None, None]) for _ in range(n)],
        "auto_renew": [random.choice([True, False]) for _ in range(n)]
    })
    
    # ── Inject Anomalies ──
    # 1. Total charges math discrepancy (~140 rows)
    bad_math = random.sample(range(n), 140)
    for idx in bad_math:
        df.at[idx, "total_charges"] = round(df.at[idx, "total_charges"] + random.uniform(50.0, 500.0), 2)
        
    # 2. Sentinel strings in total_charges (~60 rows)
    df["total_charges"] = df["total_charges"].astype(object)
    bad_totals = random.sample(range(n), 60)
    for idx in bad_totals:
        df.at[idx, "total_charges"] = random.choice(["pending", "unknown", "N/A"])
        
    # 3. Exact duplicates (~150 rows)
    dupes = df.iloc[random.sample(range(n), 150)].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    path = os.path.join(OUT_DIR, "saas_churn_metrics.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df):,} rows)")
    return path


if __name__ == "__main__":
    generate_ecommerce_orders()
    generate_healthcare_patients()
    generate_fintech_transactions()
    generate_bikeshare_trips()
    generate_hr_employees()
    generate_saas_subscriptions()
    print("\nAll 6 benchmark datasets generated successfully in ./benchmark_data/")

