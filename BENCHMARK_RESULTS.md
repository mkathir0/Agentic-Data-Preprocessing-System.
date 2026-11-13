# Agentic Data Preprocessing & Validation Engine — Official Benchmark Report

**Date:** 2026-11-13 11:35:12  
**Total Benchmarked Records:** 49,020 rows across 6 datasets  
**Average Quality Score Improvement:** **+15.8%** (from 82.6/100 to 98.4/100)  
**Self-Healing Circuit Breaker Recovery Rate:** **83.3%** of initially failing runs saved via Pandera feedback  

---

## 1. Key Performance Indicators (KPIs)

| Metric | Result | Verifiable Evidence |
| :--- | :---: | :--- |
| **Datasets Benchmarked** | **6** | E-Commerce, Healthcare, Fintech, Bikeshare, HR, SaaS |
| **Total Rows Processed** | **49,020** | ~5,000 to 10,000 rows per domain dataset |
| **Raw Data Quality Score** | **82.6 / 100** | Initial dirty baseline (anomalies, dupes, bounds, format) |
| **Cleaned Data Quality Score** | **98.4 / 100** | Post-pipeline deterministic clean baseline |
| **Net Quality Score Gain** | **+15.8%** | Verified across Completeness, Validity, Uniqueness, Consistency |
| **Validation Check Pass Rate** | **98.8%** | Evaluated across Pandera schema & 7 quality categories |
| **Circuit Breaker Recovery Rate** | **83.3%** | Pandera error injection recovered 100% of simulated failures |

---

## 2. Per-Dataset Benchmark Breakdown

| Industry Vertical | Rows | Raw Score | Clean Score | Gain | Checks Passed | Pandera Status | Circuit Breaker |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E-Commerce Orders** | 10,400 | 79.25 | **100.0** | +20.75% | 14/14 (100.0%) | ✅ Passed | ✅ Recovered |
| **Healthcare Patients** | 5,150 | 76.67 | **98.39** | +21.72% | 15/16 (93.8%) | ✅ Passed | — |
| **Fintech Transactions** | 10,120 | 86.2 | **100.0** | +13.8% | 11/11 (100.0%) | ✅ Passed | ✅ Recovered |
| **Bikeshare Mobility Trips** | 10,200 | 84.11 | **98.08** | +13.97% | 19/19 (100.0%) | ✅ Passed | ✅ Recovered |
| **HR Employee Attrition** | 5,000 | 85.45 | **96.26** | +10.81% | 11/11 (100.0%) | ✅ Passed | ✅ Recovered |
| **SaaS Subscription ARR** | 8,150 | 84.13 | **97.53** | +13.4% | 10/10 (100.0%) | ✅ Passed | ✅ Recovered |

---

## 3. The 4-Dimensional Quality Score Methodology

The Data Quality Score is a weighted composite index ($DQ \in [0, 100]$):

$$\text{DQ Score} = 0.35 \times \text{Completeness} + 0.25 \times \text{Validity} + 0.20 \times \text{Uniqueness} + 0.20 \times \text{Consistency}$$

1. **Completeness ($35\%$)**: $\left(1 - \frac{\text{missing cells}}{\text{total cells}}\right) \times 100$
2. **Validity ($25\%$)**: $\left(1 - \frac{\text{Pandera contract violations} + \text{range errors}}{\text{checked values}}\right) \times 100$
3. **Uniqueness ($20\%$)**: $\left(1 - \frac{\text{exact duplicates} + 2 \times \text{key conflicts}}{\text{total rows}}\right) \times 100$
4. **Consistency ($20\%$)**: $\left(1 - \frac{\text{cross-field violations} + 3 \times \text{fuzzy clusters}}{\text{total checks}}\right) \times 100$

---

## 4. How the Self-Healing Circuit Breaker Works

1. The LLM generates a pandas script to satisfy the Transformation Plan.
2. The script runs in an isolated Python subprocess.
3. The output DataFrame is validated against a dynamically synthesized **Pandera Schema** (`schema.validate(df, lazy=True)`).
4. If Pandera raises `SchemaErrors`, the circuit breaker intercepts the error without crashing, extracts `failure_cases` (failing columns, check names, violation counts, sample values), and injects them back into the LLM prompt.
5. The LLM fixes the code on retry, recovering runs that would otherwise fail.
