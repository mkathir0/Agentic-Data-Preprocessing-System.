"""
Standalone Verification Script for the Enterprise Data Quality & Validation Engine.
Tests all 7 validation categories, Pandera contract enforcement, fuzzy deduplication,
key-conflict detection, and before/after Data Quality scoring.
"""

import os
import pandas as pd
import numpy as np

from app.agents.validation_engine import (
    build_dynamic_pandera_schema,
    validate_pandera_contract,
    detect_fuzzy_duplicates,
    detect_key_conflicts,
    check_cross_field_relations,
    calculate_data_quality_metrics,
    execute_full_validation_engine
)
from app.agents.schemas import SchemaOutput

def run_tests():
    print("=" * 70)
    print("  VERIFYING ENTERPRISE DATA QUALITY & VALIDATION ENGINE")
    print("=" * 70)

    # ── Test 1: Pandera Contract Validation ──────────────────────────────────
    print("\n[TEST 1] Dynamic Pandera Schema Synthesis & Lazy Validation")
    test_df = pd.DataFrame({
        "user_id": [101, 102, 103, None, 105],              # PK has a null!
        "user_age": [25, 42, -5, 140, 31],                  # Age has out-of-bounds (-5, 140)
        "email_address": ["alice@work.com", "bob.invalid", "carol@test.org", "dan@co.io", "eve@web.com"], # Invalid email
        "account_balance": [150.0, -20.0, 300.5, 0.0, 50.0] # Negative balance
    })
    schema_info = SchemaOutput(
        primary_keys=["user_id"],
        candidate_foreign_keys=[],
        categorical_columns=[],
        datetime_columns=[],
        inferred_types={"user_id": "integer", "user_age": "integer", "email_address": "email", "account_balance": "float"}
    )
    pa_schema = build_dynamic_pandera_schema(test_df, schema_info)
    passed, failures, feedback, checks = validate_pandera_contract(test_df, pa_schema)

    print(f"  Pandera Passed: {passed} (Expected: False)")
    print(f"  Failure Cases Caught: {len(failures)} columns/checks failed")
    for f in failures:
        print(f"    - Column '{f['column']}': check '{f['check']}' ({f['violation_count']} violations, samples: {f['sample_failures']})")
    assert not passed, "Pandera should have failed on dirty test data!"
    assert len(failures) >= 3, "Pandera should have caught user_id null, age bounds, and email format!"
    print("  --> TEST 1 PASSED: Pandera contract correctly flagged violations.")

    # ── Test 2: Fuzzy Deduplication & Entity Resolution ──────────────────────
    print("\n[TEST 2] Fuzzy Deduplication & Entity Clustering")
    fuzzy_df = pd.DataFrame({
        "city": [
            "San Francisco", "San Fransisco", "San Francisco", "New York",
            "New York City", "Chicago", "Chicagoo", "Austin"
        ]
    })
    clusters_count, examples, fuzzy_checks = detect_fuzzy_duplicates(fuzzy_df, threshold=0.82)
    print(f"  Fuzzy Clusters Detected: {clusters_count}")
    print(f"  Cluster Examples: {examples}")
    assert clusters_count >= 2, f"Expected at least 2 clusters (SF and Chicago), got {clusters_count}"
    print("  --> TEST 2 PASSED: Fuzzy entity near-duplicates correctly grouped.")

    # ── Test 3: Primary Key Conflict vs Exact Duplicate Detection ────────────
    print("\n[TEST 3] Key Conflict Detection vs Exact Row Duplicates")
    pk_df = pd.DataFrame({
        "order_id": [1001, 1002, 1002, 1003, 1004, 1004],
        "customer": ["Alice", "Bob", "Bob", "Charlie", "David", "Diana"], # 1002 is exact dupe, 1004 is key conflict!
        "total": [25.0, 50.0, 50.0, 75.0, 100.0, 120.0]
    })
    exact_dups, key_confs, conf_details, pk_checks = detect_key_conflicts(pk_df, primary_keys=["order_id"])
    print(f"  Exact Duplicates Count: {exact_dups} (Expected: 1)")
    print(f"  Key Conflicts (PK Collisions): {key_confs} (Expected: 2)")
    print(f"  Conflict Details: {conf_details}")
    assert exact_dups == 1, f"Expected 1 exact duplicate, got {exact_dups}"
    assert key_confs == 2, f"Expected 2 rows in key collision, got {key_confs}"
    print("  --> TEST 3 PASSED: Exact duplicates correctly isolated from primary key conflicts.")

    # ── Test 4: Cross-Field Relational Invariants ─────────────────────────────
    print("\n[TEST 4] Cross-Field Relational & Temporal Ordering")
    cf_df = pd.DataFrame({
        "started_at": ["2026-01-01 10:00:00", "2026-01-01 12:00:00", "2026-01-01 14:00:00"],
        "ended_at":   ["2026-01-01 10:30:00", "2026-01-01 11:30:00", "2026-01-01 15:00:00"], # Row 1 has end < start!
        "subtotal":   [10.0, 20.0, 30.0],
        "tax":        [1.0, 2.0, 3.0],
        "total":      [11.0, 99.0, 33.0]                                                        # Row 1 has total != subtotal + tax
    })
    checks, biz_violations, patterns = check_cross_field_relations(cf_df)
    temporal_violations = sum(v.violation_count for v in biz_violations if ">=" in v.rule)
    math_violations = sum(v.violation_count for v in biz_violations if "+" in v.rule)
    print(f"  Temporal Inversions: {temporal_violations} (Expected: 1)")
    print(f"  Math Identity Violations: {math_violations} (Expected: 1)")
    assert temporal_violations == 1, "Failed to catch inverted temporal dates!"
    assert math_violations == 1, "Failed to catch arithmetic invariant failure!"
    print("  --> TEST 4 PASSED: Cross-field temporal and math rules verified.")

    # ── Test 5: Full End-to-End Validation Engine with DQ Scoring ────────────
    print("\n[TEST 5] Full End-to-End Validation Engine & Quality Scoring")
    os.makedirs("test_data", exist_ok=True)
    raw_path = "test_data/raw_sample.csv"
    clean_path = "test_data/clean_sample.csv"

    # Raw dirty dataset
    raw_df = pd.DataFrame({
        "order_id": [101, 102, 102, 103, 104, 105, 106, 107, 108, 109],
        "customer_city": ["New York", "NYC", "New York", "Chicago", "Chicagoo", "Dallas", "Houston", "Miami", "Denver", "Seattle"],
        "order_amount": [50.0, -10.0, -10.0, 100.0, 75.0, 200.0, None, 45.0, 120.0, 80.0],
        "order_date": ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"],
        "ship_date":  ["2026-01-03", "2026-01-01", "2026-01-01", "2026-01-05", "2026-01-06", "2026-01-08", "2026-01-09", "2026-01-10", "2026-01-11", "2026-01-12"],
    })
    raw_df.to_csv(raw_path, index=False)

    # Cleaned dataset (fixed anomalies)
    clean_df = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106, 107, 108, 109],
        "customer_city": ["New York", "New York", "Chicago", "Chicago", "Dallas", "Houston", "Miami", "Denver", "Seattle"],
        "order_amount": [50.0, 10.0, 100.0, 75.0, 200.0, 85.0, 45.0, 120.0, 80.0],
        "order_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"],
        "ship_date":  ["2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-08", "2026-01-09", "2026-01-10", "2026-01-11", "2026-01-12"],
    })
    clean_df.to_csv(clean_path, index=False)

    schema = SchemaOutput(
        primary_keys=["order_id"],
        candidate_foreign_keys=[],
        categorical_columns=["customer_city"],
        datetime_columns=["order_date", "ship_date"],
        inferred_types={"order_id": "integer", "customer_city": "text", "order_amount": "float", "order_date": "datetime", "ship_date": "datetime"}
    )

    report = execute_full_validation_engine(raw_path, clean_path, schema=schema)

    print(f"  Raw Data Quality Score:     {report.data_quality_score_before:.1f} / 100")
    print(f"  Cleaned Data Quality Score: {report.data_quality_score_after:.1f} / 100")
    print(f"  Quality Improvement:        +{report.score_improvement:.1f}%")
    print(f"  Pandera Contract Passed:    {report.pandera_schema_passed}")
    print(f"  Total Checks Passed:        {report.total_passed} / {report.total_checks}")

    assert report.data_quality_score_after > report.data_quality_score_before, "Cleaned data score must improve!"
    assert report.pandera_schema_passed, "Cleaned data should pass Pandera contract!"
    print("  --> TEST 5 PASSED: Full Validation Engine calculated verifiable metrics successfully!")

    # ── Test 6: Great Expectations Suite Interoperability ────────────────────
    print("\n[TEST 6] Great Expectations Interoperability & Expectation Suite Export")
    from app.agents.validation_engine import export_great_expectations_suite
    gx_suite = export_great_expectations_suite(pa_schema, suite_name="ecommerce_gx_suite")
    expectations_list = gx_suite.get("expectations", [])
    print(f"  GX Expectation Suite Name: '{gx_suite['expectation_suite_name']}'")
    print(f"  Exported Expectations Count: {len(expectations_list)}")
    sample_types = [e["expectation_type"] for e in expectations_list[:4]]
    print(f"  Sample Expectation Types: {sample_types}")
    assert len(expectations_list) >= 4, "Should have exported multiple Great Expectations assertions!"
    print("  --> TEST 6 PASSED: Pandera schema successfully mapped to Great Expectations suite.")

    print("\n" + "=" * 70)
    print("  ALL 6 VERIFICATION TESTS PASSED SUCCESSFULLY! (100% VERIFIABLE)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()

