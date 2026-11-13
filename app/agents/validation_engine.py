"""
Enterprise-Grade Data Quality & Validation Engine

Combines:
1. Pandera Contract & Dynamic Schema Validation (lazy=True failure cases)
2. 7-Category Data Quality Invariant Checking
3. Cross-Field Relational & Temporal Verification
4. Fuzzy Deduplication & Entity Resolution
5. Primary Key Collision vs Exact Duplicate Detection
6. 4-Dimensional Data Quality Scoring (0-100: Completeness, Validity, Uniqueness, Consistency)
7. Self-Healing Circuit Breaker Diagnostics & Human Escalation Rules
"""

import re
import math
import difflib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set

import pandas as pd
import numpy as np
import pandera.pandas as pa
from langsmith import traceable

from app.agents.schemas import (
    ValidationReport, ValidationCheck, EscalationFlag,
    ColumnStats, StatsDiff, MissingnessPattern, BusinessRuleViolation,
    SchemaOutput
)
from app.utils.data_loader import load_dataframe

logger = logging.getLogger(__name__)

# ── Domain Range Rules for Dynamic Pandera & Heuristics ───────────────────────
DOMAIN_RANGE_RULES = [
    (re.compile(r'\blat\b|latitude|_lat', re.I), -90.0, 90.0, "latitude"),
    (re.compile(r'\blng\b|\blon\b|longitude|_lng|_lon', re.I), -180.0, 180.0, "longitude"),
    (re.compile(r'\bage\b|_age|age_', re.I), 0.0, 120.0, "age"),
    (re.compile(r'\bbmi\b', re.I), 10.0, 70.0, "bmi"),
    (re.compile(r'percent|pct|rate', re.I), 0.0, 100.0, "percentage"),
    (re.compile(r'price|cost|amount|revenue|salary|fee|fare|subtotal|total', re.I), 0.0, None, "monetary"),
    (re.compile(r'duration|seconds|minutes|hours', re.I), 0.0, None, "duration"),
    (re.compile(r'quantity|qty|units|count|items', re.I), 0.0, None, "quantity"),
]

FORMAT_PATTERNS = {
    'email': re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'),
    'phone': re.compile(r'^\+?[\d\s\-().]{7,20}$'),
    'zip': re.compile(r'^\d{5}(-\d{4})?$'),
    'url': re.compile(r'^https?://'),
    'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pct(count: int, total: int) -> float:
    return round(count / total, 6) if total > 0 else 0.0

def _check(category: str, name: str, passed: bool, violations: int = 0, details: str = "") -> ValidationCheck:
    return ValidationCheck(
        category=category,
        name=name,
        passed=passed,
        violations=violations,
        details=details
    )

def _col_stats(df: pd.DataFrame, col: str) -> ColumnStats:
    series = df[col]
    null_count = int(series.isnull().sum())
    n = len(series)
    stats = ColumnStats(
        column=col,
        dtype=str(series.dtype),
        null_pct=_pct(null_count, n),
        unique_count=int(series.nunique()),
    )
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        clean = series.dropna()
        if len(clean) > 0:
            stats.mean = float(clean.mean())
            stats.median = float(clean.median())
            stats.std = float(clean.std()) if len(clean) > 1 else 0.0
            stats.skew = float(clean.skew()) if len(clean) > 2 else 0.0
            stats.min = float(clean.min())
            stats.max = float(clean.max())
            # IQR Outliers
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            stats.outlier_count_iqr = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
            if stats.std and stats.std > 0:
                z = (clean - stats.mean) / stats.std
                stats.outlier_count_zscore = int((z.abs() > 3).sum())
    return stats


# ── 1. Dynamic Pandera Schema Synthesis & Validation ─────────────────────────

def build_dynamic_pandera_schema(
    df: pd.DataFrame,
    schema_info: Optional[SchemaOutput] = None,
    pks_override: Optional[List[str]] = None
) -> pa.DataFrameSchema:
    """
    Synthesizes a Pandera DataFrameSchema dynamically from the inferred schema
    and domain heuristics.
    """
    columns: Dict[str, pa.Column] = {}
    pks = list(pks_override) if pks_override is not None else (list(schema_info.primary_keys) if schema_info and schema_info.primary_keys else [])

    # Detect fallback PKs if not provided
    if not pks:
        for col in df.columns:
            if re.search(r'\bid\b|_id$|uuid', col, re.I):
                if df[col].nunique() / max(len(df), 1) > 0.90:
                    pks.append(col)

    for col in df.columns:
        series = df[col]
        checks = []
        is_pk = col in pks
        nullable = not is_pk

        # Inferred target dtype
        target_type_str = schema_info.inferred_types.get(col, "").lower() if (schema_info and schema_info.inferred_types) else ""
        is_num = (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)) or ("int" in target_type_str) or ("float" in target_type_str)

        # Check domain range bounds
        if is_num:
            for pattern, lo, hi, domain in DOMAIN_RANGE_RULES:
                if pattern.search(col):
                    if lo is not None and hi is not None:
                        checks.append(pa.Check.in_range(lo, hi, name=f"{domain}_in_range_{lo}_{hi}"))
                    elif lo is not None:
                        checks.append(pa.Check.greater_than_or_equal_to(lo, name=f"{domain}_gte_{lo}"))
                    elif hi is not None:
                        checks.append(pa.Check.less_than_or_equal_to(hi, name=f"{domain}_lte_{hi}"))
                    break

        # Check regex format patterns
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            col_lower = col.lower()
            for fmt_name, pattern in FORMAT_PATTERNS.items():
                if fmt_name in col_lower:
                    checks.append(pa.Check.str_matches(pattern.pattern, name=f"valid_{fmt_name}_format"))
                    break

        # Build Pandera column contract
        if "int" in target_type_str:
            col_dtype = float if series.isnull().any() else int
        elif "float" in target_type_str or "numeric" in target_type_str:
            col_dtype = float
        elif "datetime" in target_type_str:
            col_dtype = "datetime64[ns]" if pd.api.types.is_datetime64_any_dtype(series) else object
        elif "bool" in target_type_str:
            col_dtype = bool
        else:
            col_dtype = series.dtype
        # If numeric but has nulls in pandas < 2.0 or legacy, pandas uses float
        if is_pk:
            columns[col] = pa.Column(
                col_dtype,
                checks=checks,
                nullable=False,
                unique=True,
                name=col
            )
        else:
            columns[col] = pa.Column(
                col_dtype,
                checks=checks,
                nullable=True,
                name=col
            )

    return pa.DataFrameSchema(columns=columns, coerce=False, strict=False)


def validate_pandera_contract(
    df: pd.DataFrame,
    pandera_schema: pa.DataFrameSchema
) -> Tuple[bool, List[Dict[str, Any]], str, List[ValidationCheck]]:
    """
    Validates a DataFrame against a Pandera schema using lazy validation.
    Extracts all failure cases into structured diagnostic dictionaries and
    produces clean error feedback for LLM self-healing retries.
    """
    failure_cases: List[Dict[str, Any]] = []
    checks: List[ValidationCheck] = []
    error_feedback_lines = []

    try:
        pandera_schema.validate(df, lazy=True)
        checks.append(_check("schema", "Pandera Contract: Full Schema Conformance", True, 0, "All columns conform to Pandera schema checks"))
        return True, [], "", checks
    except pa.errors.SchemaErrors as err:
        fc_df = err.failure_cases
        grouped = fc_df.groupby(['column', 'check']).agg(
            violation_count=('index', 'count'),
            sample_failures=('failure_case', lambda s: list(s.dropna().unique()[:3]))
        ).reset_index()

        for _, row in grouped.iterrows():
            col_name = str(row['column']) if pd.notna(row['column']) else "DataFrame"
            check_name = str(row['check'])
            count = int(row['violation_count'])
            samples = [str(x) for x in row['sample_failures']]
            sample_str = ", ".join(samples)

            failure_cases.append({
                "column": col_name,
                "check": check_name,
                "violation_count": count,
                "sample_failures": samples
            })

            details = f"{count} violations for check '{check_name}' on column '{col_name}' (sample values: [{sample_str}])"
            checks.append(_check("schema", f"Pandera Check: {col_name} -> {check_name}", False, count, details))
            error_feedback_lines.append(f"- Column '{col_name}': Failed check '{check_name}' on {count} rows (e.g. {sample_str})")

        formatted_feedback = (
            "PANDERA SCHEMA ERRORS (Self-Healing Circuit Breaker Triggered):\n" +
            "\n".join(error_feedback_lines) +
            "\nFix these specific column types, null constraints, or check violations in your pandas cleaning code."
        )

        return False, failure_cases, formatted_feedback, checks


# ── 2. Fuzzy Deduplication & Entity Resolution ───────────────────────────────

def detect_fuzzy_duplicates(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.85
) -> Tuple[int, List[str], List[ValidationCheck]]:
    """
    Detects near-duplicate entity strings in categorical columns using sequence matching
    and token similarity (e.g. 'San Francisco' vs 'San Fransisco', 'Google Inc' vs 'Google, Inc.').
    """
    checks = []
    cluster_examples: List[str] = []
    total_clusters = 0

    target_cols = columns if columns else df.select_dtypes(include=['object', 'string']).columns

    for col in target_cols:
        series = df[col].dropna().astype(str)
        n_unique = series.nunique()
        # Evaluate on columns with moderate cardinality (e.g. cities, departments, categories)
        if n_unique < 2 or n_unique > 300:
            continue

        unique_vals = sorted(series.unique(), key=lambda s: len(s), reverse=True)
        visited = set()
        col_clusters = []

        for i, val_a in enumerate(unique_vals):
            if val_a in visited:
                continue
            cluster = [val_a]
            norm_a = val_a.strip().lower()
            tokens_a = set(re.findall(r'\w+', norm_a))

            for val_b in unique_vals[i+1:]:
                if val_b in visited:
                    continue
                norm_b = val_b.strip().lower()
                if norm_a == norm_b:
                    continue

                # If strings differ only by numbers (e.g. 'Station 1' vs 'Station 2'), do not cluster as typos
                digits_a = re.findall(r'\d+', val_a)
                digits_b = re.findall(r'\d+', val_b)
                if digits_a and digits_b and digits_a != digits_b:
                    continue

                # Character sequence ratio
                char_sim = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

                # Token Jaccard ratio
                tokens_b = set(re.findall(r'\w+', norm_b))
                tok_sim = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if (tokens_a | tokens_b) else 0.0

                sim = max(char_sim, tok_sim)
                if sim >= threshold:
                    cluster.append(val_b)
                    visited.add(val_b)

            if len(cluster) > 1:
                col_clusters.append(cluster)
                visited.add(val_a)

        if col_clusters:
            total_clusters += len(col_clusters)
            sample_cluster = col_clusters[0]
            example_str = f"'{col}': {sample_cluster[0]} ≈ {sample_cluster[1]}"
            cluster_examples.append(example_str)
            checks.append(_check(
                "duplicates",
                f"'{col}' has consistent entity spelling (fuzzy deduplication)",
                passed=False,
                violations=len(col_clusters),
                details=f"Found {len(col_clusters)} fuzzy near-duplicate clusters (e.g. {sample_cluster})"
            ))
        else:
            checks.append(_check(
                "duplicates",
                f"'{col}' has consistent entity spelling (fuzzy deduplication)",
                passed=True,
                violations=0,
                details=f"No near-duplicate entity variations detected among {n_unique} unique values"
            ))

    return total_clusters, cluster_examples, checks


# ── 3. Key-Conflict Detection (vs Exact Duplicates) ──────────────────────────

def detect_key_conflicts(
    df: pd.DataFrame,
    primary_keys: List[str]
) -> Tuple[int, int, List[str], List[ValidationCheck]]:
    """
    Distinguishes exact full-row duplicates from Primary Key collisions.
    A PK collision (key conflict) occurs when two rows share the same PK, but
    differ in other column values (e.g. conflicting transaction amounts or names).
    """
    checks = []
    exact_duplicates = int(df.duplicated().sum())
    total_key_conflicts = 0
    conflict_details: List[str] = []

    checks.append(_check(
        "duplicates",
        "No exact duplicate rows",
        passed=(exact_duplicates == 0),
        violations=exact_duplicates,
        details=f"{exact_duplicates} fully identical rows found" if exact_duplicates else "Zero exact duplicate rows"
    ))

    for pk in primary_keys:
        if pk not in df.columns:
            continue
        pk_dup_mask = df.duplicated(subset=[pk], keep=False)
        exact_dup_mask = df.duplicated(keep=False)
        conflict_mask = pk_dup_mask & ~exact_dup_mask
        conflict_count = int(conflict_mask.sum())

        if conflict_count > 0:
            total_key_conflicts += conflict_count
            sample_keys = [str(k) for k in df.loc[conflict_mask, pk].unique()[:3]]
            sample_str = ", ".join(sample_keys)
            msg = f"{conflict_count} rows share identical primary key '{pk}' with conflicting data in other columns (keys: {sample_str})"
            conflict_details.append(msg)
            checks.append(_check(
                "duplicates",
                f"No primary key conflicts on '{pk}'",
                passed=False,
                violations=conflict_count,
                details=msg
            ))
        else:
            checks.append(_check(
                "duplicates",
                f"No primary key conflicts on '{pk}'",
                passed=True,
                violations=0,
                details=f"Primary key '{pk}' has zero key-conflict collisions"
            ))

    return exact_duplicates, total_key_conflicts, conflict_details, checks


# ── 4. Cross-Field Relational & Temporal Verification ────────────────────────

def check_cross_field_relations(
    df: pd.DataFrame,
    schema: Optional[SchemaOutput] = None
) -> Tuple[List[ValidationCheck], List[BusinessRuleViolation], List[MissingnessPattern]]:
    """
    Checks temporal ordering (start <= end), arithmetic invariants (A + B = C, A * B = C),
    and conditional structural missingness.
    """
    checks: List[ValidationCheck] = []
    violations: List[BusinessRuleViolation] = []
    patterns: List[MissingnessPattern] = []
    n = len(df)
    if n == 0:
        return checks, violations, patterns

    # 1. Temporal Ordering
    start_patterns = re.compile(r'start|begin|created|order_date|opened|from', re.I)
    end_patterns = re.compile(r'end|finish|closed|ship_date|resolved|to\b|stop|delivery_date', re.I)
    date_hint = re.compile(r'date|time|\bat\b|_at$|_on$', re.I)
    indicator_pattern = re.compile(r'missing|indicator|flag|isna|_is_|_has_|^is_|^has_', re.I)

    date_cols = [c for c in df.columns if date_hint.search(c) and not indicator_pattern.search(c)]
    for sc in date_cols:
        if not start_patterns.search(sc):
            continue
        for ec in date_cols:
            if not end_patterns.search(ec) or sc == ec:
                continue
            # Compare
            try:
                s = pd.to_datetime(df[sc], errors='coerce')
                e = pd.to_datetime(df[ec], errors='coerce')
                valid = s.notna() & e.notna()
                min_valid = min(n, 2)
                if valid.sum() < min_valid:
                    continue
                inverted = valid & (e < s)
                inv_count = int(inverted.sum())

                sample_row = None
                if inv_count > 0:
                    idx = inverted[inverted].index[0]
                    sample_row = f"Row {idx}: {sc}={df[sc].iloc[idx]} > {ec}={df[ec].iloc[idx]}"

                checks.append(_check(
                    "cross_field",
                    f"Temporal Precedence: '{ec}' >= '{sc}'",
                    passed=(inv_count == 0),
                    violations=inv_count,
                    details=f"{inv_count} rows where end timestamp precedes start timestamp" if inv_count else f"All valid rows satisfy {ec} >= {sc}"
                ))

                if inv_count > 0:
                    violations.append(BusinessRuleViolation(
                        rule=f"{ec} >= {sc}",
                        violation_count=inv_count,
                        violation_pct=_pct(inv_count, n),
                        disposition="flagged",
                        example=sample_row
                    ))
            except Exception:
                pass

    # 2. Arithmetic Invariants (e.g. subtotal + tax = total, qty * price = line_total)
    num_cols = [
        c for c in df.columns 
        if pd.api.types.is_numeric_dtype(df[c]) 
        and not pd.api.types.is_bool_dtype(df[c])
        and not indicator_pattern.search(c)
        and not re.search(r'\bid\b|_id$|uuid|lat|lng|lon|rating|zip|postal', c, re.I)
        and df[c].nunique() > 10
    ]
    if len(num_cols) >= 3:
        sample_df = df[num_cols].dropna()
        min_sample = min(n, 10)
        if len(sample_df) >= min_sample:
            for i, c1 in enumerate(num_cols):
                for j, c2 in enumerate(num_cols):
                    if i >= j:
                        continue
                    for c3 in num_cols:
                        if c3 in (c1, c2):
                            continue

                        valid_mask = df[c1].notna() & df[c2].notna() & df[c3].notna()
                        v_count = valid_mask.sum()
                        if v_count < min_sample:
                            continue

                        # Ensure columns are not dominated by zeros
                        non_zero_1 = (df.loc[valid_mask, c1] != 0).mean()
                        non_zero_2 = (df.loc[valid_mask, c2] != 0).mean()
                        if non_zero_1 < 0.2 or non_zero_2 < 0.2:
                            continue

                        # Addition check: c1 + c2 = c3 (must hold on at least 85% of data to be a real law)
                        add_close = np.isclose(df.loc[valid_mask, c1] + df.loc[valid_mask, c2], df.loc[valid_mask, c3], rtol=1e-2, atol=1e-2)
                        add_rate = add_close.sum() / v_count
                        if 0.85 <= add_rate < 0.999:
                            fail_count = int((~add_close).sum())
                            checks.append(_check(
                                "cross_field",
                                f"Arithmetic Invariant: {c1} + {c2} = {c3}",
                                passed=False,
                                violations=fail_count,
                                details=f"{fail_count} rows violate mathematical identity {c1} + {c2} = {c3}"
                            ))
                            violations.append(BusinessRuleViolation(
                                rule=f"{c1} + {c2} = {c3}",
                                violation_count=fail_count,
                                violation_pct=_pct(fail_count, n),
                                disposition="flagged",
                                example=f"{fail_count} rows with arithmetic discrepancy"
                            ))

                        # Multiplication check: c1 * c2 = c3 (must hold on at least 85% of data to be a real law)
                        mul_close = np.isclose(df.loc[valid_mask, c1] * df.loc[valid_mask, c2], df.loc[valid_mask, c3], rtol=1e-2, atol=1e-2)
                        mul_rate = mul_close.sum() / v_count
                        if 0.85 <= mul_rate < 0.999:
                            fail_count = int((~mul_close).sum())
                            checks.append(_check(
                                "cross_field",
                                f"Arithmetic Invariant: {c1} * {c2} = {c3}",
                                passed=False,
                                violations=fail_count,
                                details=f"{fail_count} rows violate mathematical identity {c1} * {c2} = {c3}"
                            ))
                            violations.append(BusinessRuleViolation(
                                rule=f"{c1} * {c2} = {c3}",
                                violation_count=fail_count,
                                violation_pct=_pct(fail_count, n),
                                disposition="flagged",
                                example=f"{fail_count} rows with arithmetic discrepancy"
                            ))

    # 3. Structural Missingness Patterns (Conditional Nulls)
    null_matrix = df.isnull()
    for target_col in df.columns:
        null_count = null_matrix[target_col].sum()
        if null_count < max(5, int(0.02 * n)):
            continue
        for cond_col in df.columns:
            if target_col == cond_col or df[cond_col].nunique() > 15:
                continue
            for val, count in df[cond_col].value_counts().items():
                if count < max(5, int(0.02 * n)):
                    continue
                mask = (df[cond_col] == val)
                target_nulls_cond = null_matrix.loc[mask, target_col].sum()
                prob = target_nulls_cond / count
                if prob > 0.95 and target_nulls_cond / null_count > 0.5:
                    patterns.append(MissingnessPattern(
                        columns=[target_col, cond_col],
                        pattern_type="structural",
                        co_null_pct=round(float(prob), 4),
                        description=(
                            f"Column '{target_col}' is missing in {round(prob*100, 1)}% of rows where "
                            f"'{cond_col}' == '{val}'. This indicates structural/conditional missingness, "
                            f"not random data loss."
                        )
                    ))

    return checks, violations, patterns


# ── 5. Data Quality Scoring Engine (0 to 100) ─────────────────────────────────

def calculate_data_quality_metrics(
    df: pd.DataFrame,
    checks: List[ValidationCheck],
    pandera_failures: int = 0,
    key_conflicts: int = 0,
    fuzzy_clusters: int = 0
) -> Dict[str, float]:
    """
    Computes a mathematical 4-dimensional Data Quality Score (0 to 100):
    DQ = 0.35 * Completeness + 0.25 * Validity + 0.20 * Uniqueness + 0.20 * Consistency
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    total_cells = max(n_rows * n_cols, 1)

    # 1. Completeness: % of non-null cells
    missing_cells = int(df.isnull().sum().sum())
    completeness = max(0.0, min(100.0, 100.0 * (1.0 - (missing_cells / total_cells))))

    # 2. Validity: % of checks passed & absence of Pandera contract failures
    total_checks = len(checks)
    failed_checks = sum(1 for c in checks if not c.passed)
    check_pass_rate = (total_checks - failed_checks) / total_checks if total_checks > 0 else 1.0
    pandera_penalty = min(0.40, (pandera_failures / max(n_rows, 1)) * 0.5)
    validity = max(0.0, min(100.0, 100.0 * (check_pass_rate - pandera_penalty)))

    # 3. Uniqueness: absence of exact duplicates and key-conflict collisions
    exact_dupes = int(df.duplicated().sum())
    dup_ratio = (exact_dupes + 2.0 * key_conflicts) / max(n_rows, 1)
    uniqueness = max(0.0, min(100.0, 100.0 * (1.0 - dup_ratio)))

    # 4. Consistency: absence of cross-field violations & fuzzy naming clusters
    cf_violations = sum(c.violations for c in checks if c.category == "cross_field")
    cf_ratio = (cf_violations + 3.0 * fuzzy_clusters) / max(n_rows, 1)
    consistency = max(0.0, min(100.0, 100.0 * (1.0 - min(cf_ratio, 1.0))))

    # Weighted Overall Score
    overall_score = round(
        0.35 * completeness +
        0.25 * validity +
        0.20 * uniqueness +
        0.20 * consistency,
        2
    )

    return {
        "overall_score": overall_score,
        "completeness": round(completeness, 2),
        "validity": round(validity, 2),
        "uniqueness": round(uniqueness, 2),
        "consistency": round(consistency, 2),
    }


# ── 6. Full 7-Category Orchestrated Validation Engine ────────────────────────

@traceable
def execute_full_validation_engine(
    input_path: str,
    output_path: str,
    schema: Optional[SchemaOutput] = None,
    profiler_before: Optional[dict] = None,
    plan_actions: Optional[List[dict]] = None
) -> ValidationReport:
    """
    Orchestrates the complete validation engine:
    - Synthesizes and checks Pandera contract on cleaned dataset
    - Evaluates 7 quality categories (Schema, Bounds, Cross-Field, Distributional, Duplicates, Formats, Regression)
    - Detects fuzzy near-duplicate entity variations
    - Detects primary key conflicts
    - Calculates Before vs After 4-dimensional Data Quality Scores
    - Formulates human escalation flags
    """
    df_before = load_dataframe(input_path)
    df_after = load_dataframe(output_path)

    report = ValidationReport(
        rows_before=len(df_before),
        rows_after=len(df_after),
        cols_before=len(df_before.columns),
        cols_after=len(df_after.columns),
        columns_before=list(df_before.columns),
        columns_after=list(df_after.columns),
        columns_dropped=[c for c in df_before.columns if c not in df_after.columns],
        columns_added=[c for c in df_after.columns if c not in df_before.columns],
    )

    # Detect candidate primary keys consistently across Pandera and collision checks
    pks = list(schema.primary_keys) if (schema and schema.primary_keys) else []
    if not pks:
        for col in df_after.columns:
            if re.search(r'\bid\b|_id$|uuid', col, re.I):
                if df_after[col].nunique() / max(len(df_after), 1) > 0.90:
                    pks.append(col)

    # ── Category 1: Schema & Pandera Contract ──
    pandera_schema = build_dynamic_pandera_schema(df_after, schema, pks_override=pks)
    pandera_passed, pandera_failures, pandera_feedback, pandera_checks = validate_pandera_contract(df_after, pandera_schema)
    report.pandera_schema_passed = pandera_passed
    report.pandera_failure_count = sum(f['violation_count'] for f in pandera_failures)
    report.pandera_failure_cases = pandera_failures
    report.schema_checks.extend(pandera_checks)

    # Dtype preservation check
    for col in df_before.columns:
        if col in df_after.columns:
            dt_b = df_before[col].dtype
            dt_a = df_after[col].dtype
            if pd.api.types.is_numeric_dtype(dt_b) and not pd.api.types.is_numeric_dtype(dt_a):
                report.schema_checks.append(_check(
                    "schema",
                    f"'{col}' dtype preserved as numeric",
                    passed=False,
                    violations=1,
                    details=f"Numeric column silently converted to {dt_a}"
                ))

    # ── Category 2: Range & Bounds ──
    for col in df_after.columns:
        if not pd.api.types.is_numeric_dtype(df_after[col]) or pd.api.types.is_bool_dtype(df_after[col]):
            continue
        if re.search(r'missing|indicator|flag|isna', col, re.I):
            continue
        series = df_after[col].dropna()
        if len(series) == 0:
            continue
        for pattern, lo, hi, domain in DOMAIN_RANGE_RULES:
            if pattern.search(col):
                below = int((series < lo).sum()) if lo is not None else 0
                above = int((series > hi).sum()) if hi is not None else 0
                tot = below + above
                report.range_checks.append(_check(
                    "range_bounds",
                    f"'{col}' within {domain} bounds [{lo}, {hi}]",
                    passed=(tot == 0),
                    violations=tot,
                    details=f"{tot} out-of-bound values" if tot else f"All values in [{lo}, {hi}]"
                ))
                break

    # ── Category 3: Cross-Field Relational ──
    cf_checks, biz_violations, miss_patterns = check_cross_field_relations(df_after, schema)
    report.cross_field_checks = cf_checks
    report.business_rule_violations = biz_violations
    report.missingness_patterns = miss_patterns

    # ── Category 4: Distributional & Imputation Shift ──
    diffs = []
    dist_checks = []
    common_cols = set(df_before.columns) & set(df_after.columns)
    for col in sorted(common_cols):
        bs = _col_stats(df_before, col)
        as_ = _col_stats(df_after, col)
        null_b = int(df_before[col].isnull().sum())
        null_a = int(df_after[col].isnull().sum())
        imputed = max(0, null_b - null_a)
        imp_pct = _pct(imputed, len(df_after))

        diffs.append(StatsDiff(
            column=col,
            before=bs,
            after=as_,
            imputation_pct=imp_pct
        ))

        # Check drastic mean shift (>20%)
        if bs.mean and abs(bs.mean) > 1e-6 and as_.mean is not None:
            shift = abs(as_.mean - bs.mean) / abs(bs.mean)
            dist_checks.append(_check(
                "distributional",
                f"'{col}' mean shift ≤ 20%",
                passed=(shift < 0.20),
                violations=1 if shift >= 0.20 else 0,
                details=f"Mean: {round(bs.mean, 2)} → {round(as_.mean, 2)} ({round(shift*100, 1)}% shift)"
            ))

        # Imputation threshold check (>15% synthetic values)
        if imp_pct > 0.15:
            dist_checks.append(_check(
                "distributional",
                f"'{col}' imputation rate ≤ 15%",
                passed=False,
                violations=imputed,
                details=f"{round(imp_pct*100, 1)}% of values are synthetically imputed"
            ))

    report.distributional_checks = dist_checks
    report.stats_diff = diffs

    # ── Category 5: Duplicates, Key Conflicts, & Fuzzy Near-Duplicates ──
    exact_dups, key_confs, conf_details, dup_checks = detect_key_conflicts(df_after, pks)
    fuzzy_clusters, fuzzy_examples, fuzzy_checks = detect_fuzzy_duplicates(df_after)

    report.duplicate_checks = dup_checks + fuzzy_checks
    report.key_based_duplicate_count = key_confs
    report.key_conflict_count = key_confs
    report.key_conflict_details = conf_details
    report.fuzzy_duplicate_clusters = fuzzy_clusters
    report.fuzzy_duplicate_examples = fuzzy_examples

    # ── Category 6: Format & Encoding ──
    format_checks = []
    for col in df_after.select_dtypes(include=['object', 'string']).columns:
        series = df_after[col].dropna()
        if len(series) == 0:
            continue
        # Whitespace
        has_ws = int((series != series.str.strip()).sum())
        if has_ws > 0:
            format_checks.append(_check(
                "format_encoding",
                f"'{col}' has no unstripped leading/trailing whitespace",
                passed=False,
                violations=has_ws,
                details=f"{has_ws} values have untrimmed whitespace"
            ))
        # Sentinels in object columns (e.g. 'N/A', '-999', 'null')
        coerced = pd.to_numeric(series, errors='coerce')
        parsed = coerced.notna().sum()
        total = len(series)
        if 0.5 * total < parsed < total:
            sentinels = int(total - parsed)
            format_checks.append(_check(
                "format_encoding",
                f"'{col}' contains sentinel strings preventing numeric casting",
                passed=False,
                violations=sentinels,
                details=f"{sentinels} sentinel strings found in otherwise numeric column"
            ))
    report.format_checks = format_checks

    # ── Category 7: Regression & Safety ──
    reg_checks = []
    new_nulls = {}
    row_drop = len(df_before) - len(df_after)
    row_drop_pct = _pct(row_drop, len(df_before)) * 100

    reg_checks.append(_check(
        "regression",
        "Row retention within safety threshold (<= 15% drop)",
        passed=(row_drop_pct <= 15.0),
        violations=max(0, row_drop),
        details=f"Rows: {len(df_before):,} → {len(df_after):,} ({row_drop:+,} rows, {round(row_drop_pct, 1)}% dropped)"
    ))

    # New nulls introduced by cleaning
    for col in df_before.columns:
        if col in df_after.columns:
            nb = int(df_before[col].isnull().sum())
            na = int(df_after[col].isnull().sum())
            if na > nb:
                intro = na - nb
                new_nulls[col] = intro
                reg_checks.append(_check(
                    "regression",
                    f"'{col}' — no new nulls introduced by cleaning",
                    passed=False,
                    violations=intro,
                    details=f"Cleaning script introduced {intro} new nulls"
                ))

    if not new_nulls:
        reg_checks.append(_check(
            "regression",
            "No new null values introduced by cleaning pipeline",
            passed=True,
            violations=0,
            details="Zero new null values introduced"
        ))

    report.regression_checks = reg_checks
    report.new_nulls_introduced = new_nulls

    # ── Totals & Overall Pass/Fail ──
    all_checks = (
        report.schema_checks + report.range_checks + report.cross_field_checks +
        report.distributional_checks + report.duplicate_checks +
        report.format_checks + report.regression_checks
    )
    report.total_checks = len(all_checks)
    report.total_passed = sum(1 for c in all_checks if c.passed)
    report.total_failed = sum(1 for c in all_checks if not c.passed)
    report.overall_passed = (report.total_failed == 0 and report.pandera_schema_passed)

    # ── 4-Dimensional Data Quality Scores ──
    # Evaluate raw dataset (df_before) against all checks to get authentic baseline score
    pre_pa_passed, pre_pa_failures, _, pre_pa_checks = validate_pandera_contract(df_before, pandera_schema)
    exact_dups_b, key_confs_b, _, dup_checks_b = detect_key_conflicts(df_before, pks)
    fuzzy_clusters_b, _, fuzzy_checks_b = detect_fuzzy_duplicates(df_before)
    pre_cf_checks, _, _ = check_cross_field_relations(df_before, schema)

    pre_range_checks = []
    for col in df_before.columns:
        if not pd.api.types.is_numeric_dtype(df_before[col]) or pd.api.types.is_bool_dtype(df_before[col]):
            continue
        series = df_before[col].dropna()
        if len(series) == 0:
            continue
        for pattern, lo, hi, domain in DOMAIN_RANGE_RULES:
            if pattern.search(col):
                below = int((series < lo).sum()) if lo is not None else 0
                above = int((series > hi).sum()) if hi is not None else 0
                tot = below + above
                pre_range_checks.append(_check("range_bounds", f"'{col}' bounds", tot == 0, tot))
                break

    all_pre_checks = pre_pa_checks + pre_range_checks + dup_checks_b + fuzzy_checks_b + pre_cf_checks
    dq_before = calculate_data_quality_metrics(
        df_before,
        all_pre_checks,
        pandera_failures=sum(f['violation_count'] for f in pre_pa_failures),
        key_conflicts=key_confs_b,
        fuzzy_clusters=fuzzy_clusters_b
    )

    dq_after = calculate_data_quality_metrics(
        df_after,
        all_checks,
        pandera_failures=report.pandera_failure_count,
        key_conflicts=report.key_conflict_count,
        fuzzy_clusters=report.fuzzy_duplicate_clusters
    )

    report.data_quality_score_before = dq_before["overall_score"]
    report.data_quality_score_after = dq_after["overall_score"]
    report.score_improvement = round(dq_after["overall_score"] - dq_before["overall_score"], 2)
    report.completeness_score = dq_after["completeness"]
    report.validity_score = dq_after["validity"]
    report.uniqueness_score = dq_after["uniqueness"]
    report.consistency_score = dq_after["consistency"]

    # ── Escalation Logic ──
    flags: List[EscalationFlag] = []

    # High Imputation
    for diff in report.stats_diff:
        if diff.imputation_pct > 0.20:
            flags.append(EscalationFlag(
                column=diff.column,
                reason=f"{round(diff.imputation_pct*100, 1)}% of values in '{diff.column}' are synthetically imputed — downstream models should note artificial distribution",
                severity="high"
            ))

    # Key Conflicts
    if report.key_conflict_count > 0:
        flags.append(EscalationFlag(
            column=pks[0] if pks else "primary_key",
            reason=f"Found {report.key_conflict_count} Primary Key collisions where identical IDs hold conflicting record data. Requires human business reconciliation.",
            severity="high"
        ))

    # Business Rule Violations
    for v in report.business_rule_violations:
        if v.violation_pct > 0.05:
            flags.append(EscalationFlag(
                column=None,
                reason=f"Cross-field rule '{v.rule}' violated by {v.violation_count} rows ({round(v.violation_pct*100, 1)}%)",
                severity="high"
            ))

    # Pandera Failures
    if not report.pandera_schema_passed:
        flags.append(EscalationFlag(
            column=None,
            reason=f"Pandera contract check failed with {report.pandera_failure_count} total violations across schema constraints.",
            severity="high"
        ))

    # New Nulls
    for col, count in report.new_nulls_introduced.items():
        flags.append(EscalationFlag(
            column=col,
            reason=f"Pipeline introduced {count} new null values in '{col}'",
            severity="high"
        ))

    report.exact_duplicates_removed = max(0, len(df_before) - len(df_after))

    # ── Category 7: Great Expectations Suite Generation ──
    try:
        report.great_expectations_suite = export_great_expectations_suite(
            pandera_schema,
            suite_name=f"quality_suite_{Path(output_path).stem}"
        )
    except Exception as e:
        logger.warning(f"Failed to export Great Expectations suite: {e}")

    report.escalation_flags = flags

    return report


# ── 7. Great Expectations Interoperability & Suite Export ────────────────────

def export_great_expectations_suite(
    pandera_schema: pa.DataFrameSchema,
    suite_name: str = "cleaned_data_quality_suite"
) -> Dict[str, Any]:
    """
    Translates Pandera DataFrameSchema checks into standard Great Expectations (GX) expectation suites.
    Enables zero-friction interoperability: validate in-memory via Pandera lazy contracts,
    and export to Great Expectations JSON format for downstream CI/CD or dbt data docs.
    """
    expectations = []

    # Table-level expectation
    expectations.append({
        "expectation_type": "expect_table_columns_to_match_ordered_set",
        "kwargs": {
            "column_list": list(pandera_schema.columns.keys())
        }
    })

    for col_name, col_obj in pandera_schema.columns.items():
        # Nullability
        if not col_obj.nullable:
            expectations.append({
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": col_name}
            })

        # Uniqueness
        if col_obj.unique:
            expectations.append({
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {"column": col_name}
            })

        # Dtype
        expectations.append({
            "expectation_type": "expect_column_values_to_be_of_type",
            "kwargs": {"column": col_name, "type_": str(col_obj.dtype)}
        })

        # Checks
        if col_obj.checks:
            for check in col_obj.checks:
                check_name = check.name or ""
                if "in_range" in check_name:
                    parts = check_name.split("_")
                    if len(parts) >= 4:
                        try:
                            lo = float(parts[-2])
                            hi = float(parts[-1])
                            expectations.append({
                                "expectation_type": "expect_column_values_to_be_between",
                                "kwargs": {"column": col_name, "min_value": lo, "max_value": hi}
                            })
                        except Exception:
                            pass
                elif "format" in check_name or "regex" in check_name or "matches" in check_name:
                    expectations.append({
                        "expectation_type": "expect_column_values_to_match_regex",
                        "kwargs": {"column": col_name, "regex": "standard_format_pattern"}
                    })

    return {
        "expectation_suite_name": suite_name,
        "meta": {
            "great_expectations_version": "0.18.0",
            "generator": "Agentic Data Preprocessing Pandera-to-GX Exporter"
        },
        "expectations": expectations
    }


