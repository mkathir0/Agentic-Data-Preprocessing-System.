# Deterministic Cleaning Policy Benchmark

Generated: 2026-11-13T12:39:24.287681+00:00

> This measures only the allowlisted whitespace and exact whole-row deduplication policy. It does not measure LLM recovery or approve data for production.

| Dataset | Rows before | Rows after | Integrity | Validation | Review required | Score delta |
|---|---:|---:|:---:|:---:|:---:|---:|
| bikeshare_trips.csv | 10200 | 10000 | True | False | True | +7.45 |
| ecommerce_orders.csv | 10400 | 10119 | True | False | True | +13.54 |
| fintech_transactions.csv | 10120 | 10120 | True | False | True | +5.96 |
| healthcare_patients.csv | 5150 | 5000 | True | False | True | +11.01 |
| hr_employee_attrition.csv | 5000 | 5000 | True | True | False | +0.00 |
| saas_churn_metrics.csv | 8150 | 8000 | True | False | True | +6.04 |
