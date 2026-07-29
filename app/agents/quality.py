"""
7-Category Data Quality Validator

1. Schema-level validation
2. Range & bounds validation
3. Cross-field / relational validation
4. Distributional / statistical validation
5. Duplicate & integrity validation
6. Format / encoding validation
7. Post-transformation regression check
"""
import re
import math
import logging
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
from langsmith import traceable

from app.agents.schemas import (
    QualityOutput, QualityAnomaly,
    ValidationReport, ValidationCheck, EscalationFlag,
    ColumnStats, StatsDiff, MissingnessPattern, BusinessRuleViolation,
    SchemaOutput
)
from app.utils.data_loader import load_dataframe

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _pct(count: int, total: int) -> float:
    return round(count / total, 6) if total > 0 else 0.0

def _check(category, name, passed, violations=0, details="") -> ValidationCheck:
    return ValidationCheck(
        category=category, name=name, passed=passed,
        violations=violations, details=details
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
    numeric = pd.api.types.is_numeric_dtype(series)
    if numeric:
        clean = series.dropna()
        if len(clean) > 0:
            stats.mean   = float(clean.mean())
            stats.median = float(clean.median())
            stats.std    = float(clean.std()) if len(clean) > 1 else 0.0
            stats.skew   = float(clean.skew()) if len(clean) > 2 else 0.0
            stats.min    = float(clean.min())
            stats.max    = float(clean.max())
            # IQR outliers
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            stats.outlier_count_iqr = int(((clean < q1 - 1.5*iqr) | (clean > q3 + 1.5*iqr)).sum())
            # Z-score outliers
            if stats.std and stats.std > 0:
                z = (clean - stats.mean) / stats.std
                stats.outlier_count_zscore = int((z.abs() > 3).sum())
    return stats


# ── Category 1: Schema-level ─────────────────────────────────────────────────

def check_schema(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    schema: Optional[SchemaOutput] = None
) -> Tuple[List[ValidationCheck], List[str], List[str], List[str]]:
    checks = []
    cols_before = list(df_before.columns)
    cols_after  = list(df_after.columns)
    dropped = [c for c in cols_before if c not in cols_after]
    added   = [c for c in cols_after  if c not in cols_before]

    # Column presence
    checks.append(_check("schema", "No columns unexpectedly dropped",
        passed=(len(dropped) == 0), violations=len(dropped),
        details=f"Dropped: {dropped}" if dropped else "All original columns preserved"))

    # Filter out columns that were intentionally created by the pipeline
    unexpected_added = [c for c in added if not c.endswith('_review') and not c.startswith('is_') and not c.startswith('has_')]
    
    checks.append(_check("schema", "No unexpected new columns added",
        passed=(len(unexpected_added) == 0), violations=len(unexpected_added),
        details=f"Unexpectedly added: {unexpected_added}" if unexpected_added else "No unexpected new columns introduced"))

    # Non-null constraints on PKs / IDs
    if schema and schema.primary_keys:
        for pk in schema.primary_keys:
            if pk in df_after.columns:
                null_count = int(df_after[pk].isnull().sum())
                checks.append(_check("schema", f"PK '{pk}' has no nulls after cleaning",
                    passed=(null_count == 0), violations=null_count,
                    details=f"{null_count} null values found in primary key column"))

    # Uniqueness on PKs
    if schema and schema.primary_keys:
        for pk in schema.primary_keys:
            if pk in df_after.columns:
                dup_count = int(df_after[pk].duplicated().sum())
                checks.append(_check("schema", f"PK '{pk}' is unique after cleaning",
                    passed=(dup_count == 0), violations=dup_count,
                    details=f"{dup_count} duplicate key values in '{pk}'"))

    # Cardinality bounds — categoricals shouldn't explode
    if schema and schema.categorical_columns:
        for col in schema.categorical_columns:
            if col in df_before.columns and col in df_after.columns:
                card_before = df_before[col].nunique()
                card_after  = df_after[col].nunique()
                exploded = card_after > card_before * 1.5 and card_after - card_before > 3
                checks.append(_check("schema", f"Cardinality of '{col}' not artificially inflated",
                    passed=not exploded, violations=max(0, card_after - card_before),
                    details=f"Unique values: {card_before} before → {card_after} after"))

    # Dtype conformity — no column that was numeric should become object
    for col in cols_before:
        if col in cols_after:
            dtype_before = df_before[col].dtype
            dtype_after  = df_after[col].dtype
            if pd.api.types.is_numeric_dtype(dtype_before) and not pd.api.types.is_numeric_dtype(dtype_after):
                checks.append(_check("schema", f"Column '{col}' dtype preserved",
                    passed=False, violations=1,
                    details=f"Numeric dtype {dtype_before} silently became {dtype_after} after cleaning"))

    return checks, cols_before, cols_after, dropped


# ── Category 2: Range & bounds ────────────────────────────────────────────────

# Heuristic domain rules based on column name patterns
RANGE_RULES = [
    (re.compile(r'\blat\b|latitude',  re.I), -90.0,  90.0,  "latitude"),
    (re.compile(r'\blng\b|\blon\b|longitude', re.I), -180.0, 180.0, "longitude"),
    (re.compile(r'\bage\b',           re.I),   0.0,  120.0, "age"),
    (re.compile(r'percent|pct|rate',  re.I),   0.0,  100.0, "percentage"),
    (re.compile(r'price|cost|amount|revenue|salary', re.I), 0.0, None, "monetary"),
    (re.compile(r'duration|seconds|minutes|hours', re.I), 0.0, None, "duration"),
]

def check_range_bounds(df: pd.DataFrame) -> List[ValidationCheck]:
    checks = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        for pattern, lo, hi, domain in RANGE_RULES:
            if pattern.search(col):
                violations = 0
                detail_parts = []
                if lo is not None:
                    below = int((series < lo).sum())
                    violations += below
                    if below: detail_parts.append(f"{below} values < {lo}")
                if hi is not None:
                    above = int((series > hi).sum())
                    violations += above
                    if above: detail_parts.append(f"{above} values > {hi}")
                checks.append(_check("range_bounds", f"'{col}' within {domain} bounds [{lo}, {hi}]",
                    passed=(violations == 0), violations=violations,
                    details="; ".join(detail_parts) if detail_parts else f"All values within [{lo}, {hi}]"))
                break

    # Date sanity
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'at' in col.lower():
            try:
                parsed = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(parsed) > 0:
                    future = int((parsed > pd.Timestamp.now()).sum())
                    ancient = int((parsed < pd.Timestamp('1900-01-01')).sum())
                    total_violations = future + ancient
                    details = []
                    if future: details.append(f"{future} future dates")
                    if ancient: details.append(f"{ancient} dates before 1900")
                    checks.append(_check("range_bounds", f"'{col}' date bounds (no future/pre-1900)",
                        passed=(total_violations == 0), violations=total_violations,
                        details="; ".join(details) if details else "All dates within expected range"))
                        
                # Check for mixed date formats
                if df[col].dtype == object:
                    s = df[col].dropna().astype(str)
                    has_slash = s.str.contains(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
                    if has_slash.any():
                        extracted = s[has_slash].str.extract(r'(\d{1,2})[/-](\d{1,2})[/-]')
                        if extracted.shape[1] == 2:
                            part1 = pd.to_numeric(extracted[0], errors='coerce')
                            part2 = pd.to_numeric(extracted[1], errors='coerce')
                            if (part1 > 12).any() and (part2 > 12).any():
                                checks.append(_check("format_encoding", f"'{col}' has mixed date formats (DD/MM vs MM/DD)",
                                    passed=False, violations=len(s),
                                    details=f"Column contains ambiguous dates that mix DD/MM and MM/DD formats, risking silent mis-parsing"))
            except Exception:
                pass

    return checks


# ── Category 3: Cross-field / Relational ─────────────────────────────────────

def _find_temporal_pairs(df: pd.DataFrame):
    """Detect start/end column pairs heuristically."""
    pairs = []
    start_patterns = re.compile(r'start|begin|created|open|from', re.I)
    end_patterns   = re.compile(r'end|finish|closed|to\b|stop', re.I)
    date_hint      = re.compile(r'date|time|\bat\b|_at$|_on$', re.I)

    start_cols = [c for c in df.columns if start_patterns.search(c) and date_hint.search(c)]
    end_cols   = [c for c in df.columns if end_patterns.search(c)   and date_hint.search(c)]

    for sc in start_cols:
        for ec in end_cols:
            # e.g. started_at / ended_at, created_at / closed_at
            sc_root = re.sub(r'start|begin|from', '', sc, flags=re.I).strip('_').lower()
            ec_root = re.sub(r'end|finish|to|stop', '', ec, flags=re.I).strip('_').lower()
            # Require the root to match (e.g. 'ed_at' == 'ed_at') to avoid false positives like 'start_lat' and 'ended_at'
            if sc_root == ec_root:
                pairs.append((sc, ec))
    return pairs

def _find_conditional_nulls(df: pd.DataFrame, patterns: List[MissingnessPattern], n: int):
    null_matrix = df.isnull()
    for target_col in df.columns:
        null_count = null_matrix[target_col].sum()
        if null_count < max(5, 0.01 * n):
            continue
            
        for cond_col in df.columns:
            if target_col == cond_col:
                continue
            # Only check categorical/low cardinality columns for conditions
            if df[cond_col].nunique() > 20:
                continue
                
            for val, count in df[cond_col].value_counts().items():
                if count < max(5, 0.01 * n):
                    continue
                
                mask = (df[cond_col] == val)
                target_nulls_in_cond = null_matrix.loc[mask, target_col].sum()
                
                prob_null_given_cond = target_nulls_in_cond / count
                prob_cond_given_null = target_nulls_in_cond / null_count
                
                if prob_null_given_cond > 0.95 and prob_cond_given_null > 0.5:
                    patterns.append(MissingnessPattern(
                        columns=[target_col, cond_col],
                        pattern_type="conditional_structural",
                        co_null_pct=round(float(prob_null_given_cond), 4),
                        description=(
                            f"'{target_col}' is missing in {round(prob_null_given_cond*100, 1)}% "
                            f"of rows where '{cond_col}' == '{val}'. This is a structural "
                            f"business rule (conditional null), not random missingness."
                        )
                    ))

def _find_math_relationships(df: pd.DataFrame, checks: List[ValidationCheck], violations: List[BusinessRuleViolation], n: int):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 3:
        return
        
    sample_df = df.sample(min(10000, n)) if n > 10000 else df
    
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            if i >= j: continue
            for c3 in num_cols:
                if c3 == c1 or c3 == c2: continue
                
                valid_mask = sample_df[c1].notna() & sample_df[c2].notna() & sample_df[c3].notna()
                valid_count = valid_mask.sum()
                if valid_count < max(10, 0.05 * n):
                    continue
                
                # Test addition
                add_match = np.isclose(sample_df[c1] + sample_df[c2], sample_df[c3], equal_nan=False)
                add_rate = add_match[valid_mask].sum() / valid_count
                if 0.5 < add_rate < 1.0:
                    full_valid = df[c1].notna() & df[c2].notna() & df[c3].notna()
                    full_add_match = np.isclose(df[c1] + df[c2], df[c3], equal_nan=False)
                    failed = full_valid & ~full_add_match
                    fail_count = int(failed.sum())
                    if fail_count > 0:
                        checks.append(_check("cross_field", f"Math Rule: {c1} + {c2} = {c3}", passed=False, violations=fail_count, details=f"{fail_count} rows violate {c1} + {c2} = {c3}"))
                        violations.append(BusinessRuleViolation(rule=f"{c1} + {c2} = {c3}", violation_count=fail_count, violation_pct=_pct(fail_count, n), disposition="flagged", example=f"{fail_count} rows violate the discovered math rule"))
                        
                # Test multiplication
                mul_match = np.isclose(sample_df[c1] * sample_df[c2], sample_df[c3], equal_nan=False)
                mul_rate = mul_match[valid_mask].sum() / valid_count
                if 0.5 < mul_rate < 1.0:
                    full_valid = df[c1].notna() & df[c2].notna() & df[c3].notna()
                    full_mul_match = np.isclose(df[c1] * df[c2], df[c3], equal_nan=False)
                    failed = full_valid & ~full_mul_match
                    fail_count = int(failed.sum())
                    if fail_count > 0:
                        checks.append(_check("cross_field", f"Math Rule: {c1} * {c2} = {c3}", passed=False, violations=fail_count, details=f"{fail_count} rows violate {c1} * {c2} = {c3}"))
                        violations.append(BusinessRuleViolation(rule=f"{c1} * {c2} = {c3}", violation_count=fail_count, violation_pct=_pct(fail_count, n), disposition="flagged", example=f"{fail_count} rows violate the discovered math rule"))

def check_cross_field(df: pd.DataFrame, schema: Optional[SchemaOutput] = None) -> Tuple[List[ValidationCheck], List[BusinessRuleViolation], List[MissingnessPattern]]:
    checks = []
    violations = []
    patterns = []
    n = len(df)
    if n == 0:
        return checks, violations, patterns

    # Temporal ordering
    for start_col, end_col in _find_temporal_pairs(df):
        try:
            s = pd.to_datetime(df[start_col], errors='coerce')
            e = pd.to_datetime(df[end_col],   errors='coerce')
            both_valid = s.notna() & e.notna()
            inverted   = (both_valid) & (e < s)
            inv_count  = int(inverted.sum())
            example = None
            if inv_count > 0:
                idx = inverted[inverted].index[0]
                example = f"Row {idx}: {start_col}={df[start_col].iloc[idx]}, {end_col}={df[end_col].iloc[idx]}"
            checks.append(_check("cross_field", f"Temporal: {end_col} >= {start_col}",
                passed=(inv_count == 0), violations=inv_count,
                details=f"{inv_count} rows where {end_col} < {start_col}"))
            if inv_count > 0:
                violations.append(BusinessRuleViolation(
                    rule=f"{end_col} >= {start_col}",
                    violation_count=inv_count,
                    violation_pct=_pct(inv_count, n),
                    disposition="flagged",
                    example=example
                ))
        except Exception:
            pass

    # Mutual null pattern detection
    null_matrix = df.isnull()
    cols = list(df.columns)
    checked_pairs = set()
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            key = (c1, c2)
            if key in checked_pairs:
                continue
            checked_pairs.add(key)
            n1 = null_matrix[c1]
            n2 = null_matrix[c2]
            both_null = (n1 & n2).sum()
            only_one  = ((n1 & ~n2) | (~n1 & n2)).sum()
            either    = (n1 | n2).sum()
            if either > 0 and both_null > 0:
                co_pct = both_null / either
                # Strong structural pattern: >90% co-null
                if co_pct > 0.9 and either > max(10, 0.01 * n):
                    patterns.append(MissingnessPattern(
                        columns=[c1, c2],
                        pattern_type="structural",
                        co_null_pct=round(float(co_pct), 4),
                        description=(
                            f"'{c1}' and '{c2}' are null together in {round(co_pct*100, 1)}% "
                            f"of cases where either is null — likely structural (same data source), "
                            f"not random. Consider treating as a category rather than imputing."
                        )
                    ))
                elif co_pct > 0.5 and either > max(10, 0.01 * n):
                    patterns.append(MissingnessPattern(
                        columns=[c1, c2],
                        pattern_type="MAR",
                        co_null_pct=round(float(co_pct), 4),
                        description=(
                            f"'{c1}' and '{c2}' co-occur null in {round(co_pct*100, 1)}% of missing cases "
                            f"— likely Missing At Random (MAR), related to the same upstream condition."
                        )
                    ))

    _find_conditional_nulls(df, patterns, n)
    _find_math_relationships(df, checks, violations, n)

    return checks, violations, patterns


# ── Category 4: Distributional ────────────────────────────────────────────────

def check_distributional(
    df_before: pd.DataFrame,
    df_after:  pd.DataFrame,
    plan_actions: Optional[List[dict]] = None
) -> Tuple[List[ValidationCheck], List[StatsDiff]]:
    checks = []
    diffs  = []
    n_before = len(df_before)
    n_after  = len(df_after)

    cols_before = set(df_before.columns)
    cols_after  = set(df_after.columns)
    common_cols = cols_before & cols_after

    # Build a lookup of which columns had fill/imputation actions
    imputed_cols = set()
    if plan_actions:
        for action in plan_actions:
            if action.get('action_type', '').startswith('fill') or 'impute' in action.get('action_type', '').lower():
                col = action.get('column')
                if col:
                    imputed_cols.add(col)

    for col in sorted(common_cols):
        before = df_before[col].dropna()
        after  = df_after[col].dropna()
        if len(before) == 0 or len(after) == 0:
            continue

        bs = _col_stats(df_before, col)
        as_ = _col_stats(df_after, col)

        # Imputation ratio: how many values in after are synthetic
        null_before = int(df_before[col].isnull().sum())
        null_after  = int(df_after[col].isnull().sum())
        imputed_count = max(0, null_before - null_after)
        imputation_pct = _pct(imputed_count, n_after)

        diffs.append(StatsDiff(
            column=col,
            before=bs,
            after=as_,
            imputation_pct=imputation_pct
        ))

        # Flag if mean shifted drastically (>20% relative change)
        if bs.mean and abs(bs.mean) > 1e-9:
            mean_shift = abs((as_.mean or 0) - bs.mean) / abs(bs.mean)
            checks.append(_check("distributional", f"'{col}' mean not drastically shifted by cleaning",
                passed=(mean_shift < 0.20),
                violations=1 if mean_shift >= 0.20 else 0,
                details=f"Mean: {round(bs.mean, 4)} → {round(as_.mean or 0, 4)} ({round(mean_shift*100, 1)}% shift)"))

        # High imputation impact warning
        if imputation_pct > 0.15:
            checks.append(_check("distributional", f"'{col}' imputation impact ≤ 15%",
                passed=False, violations=imputed_count,
                details=f"{round(imputation_pct*100, 1)}% of values are now synthetic — statistical properties partially artificial"))

        # Outlier count change
        if bs.outlier_count_iqr is not None and as_.outlier_count_iqr is not None:
            new_outliers = max(0, as_.outlier_count_iqr - bs.outlier_count_iqr)
            checks.append(_check("distributional", f"'{col}' no new IQR outliers introduced",
                passed=(new_outliers == 0), violations=new_outliers,
                details=f"IQR outliers: {bs.outlier_count_iqr} before → {as_.outlier_count_iqr} after"))

    return checks, diffs


# ── Category 5: Duplicates & Integrity ──────────────────────────────────────

def check_duplicates(df: pd.DataFrame, schema: Optional[SchemaOutput] = None) -> Tuple[List[ValidationCheck], int]:
    checks = []
    n = len(df)

    # Exact duplicates
    exact_dup = int(df.duplicated().sum())
    checks.append(_check("duplicates", "No exact duplicate rows",
        passed=(exact_dup == 0), violations=exact_dup,
        details=f"{exact_dup} fully identical rows found"))

    # Key-based duplicates: same PK, different other values
    key_dup_total = 0
    pks = []
    if schema and schema.primary_keys:
        pks = schema.primary_keys
    else:
        # Heuristic PK detection
        for col in df.columns:
            if re.search(r'\bid\b|_id$|uuid', col, re.I):
                pks.append(col)
                
    for pk in pks:
        if pk in df.columns:
            pk_dup_mask = df.duplicated(subset=[pk], keep=False)
            exact_dup_mask = df.duplicated(keep=False)
            key_conflict_mask = pk_dup_mask & ~exact_dup_mask
            key_only_dup = int(key_conflict_mask.sum())
            key_dup_total += key_only_dup
            if key_only_dup > 0:
                checks.append(_check("duplicates", f"No key-based conflicts on '{pk}'",
                    passed=False, violations=key_only_dup,
                    details=f"{key_only_dup} rows share the same '{pk}' but have conflicting data in other columns (Primary Key Conflict)"))
            else:
                checks.append(_check("duplicates", f"No key-based conflicts on '{pk}'",
                    passed=True, violations=0,
                    details=f"No Primary Key conflicts found on '{pk}'"))

    # Near-duplicate string normalisation check on object columns
    for col in df.select_dtypes(include='object').columns:
        unique_raw = df[col].dropna().nunique()
        if unique_raw < 2 or unique_raw > 200:  # Skip high-cardinality text
            continue
        normalized = df[col].dropna().str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        unique_norm = normalized.nunique()
        near_dup_groups = unique_raw - unique_norm
        if near_dup_groups > 0:
            checks.append(_check("duplicates", f"'{col}' has consistent formatting (near-duplicate check)",
                passed=False, violations=near_dup_groups,
                details=f"{near_dup_groups} value groups appear to be the same when normalised (e.g. 'USA', 'U.S.A.', 'United States')"))

    return checks, key_dup_total


# ── Category 6: Format / Encoding ────────────────────────────────────────────

FORMAT_PATTERNS = {
    'email':    re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'),
    'phone':    re.compile(r'^\+?[\d\s\-().]{7,20}$'),
    'zip':      re.compile(r'^\d{5}(-\d{4})?$'),
    'url':      re.compile(r'^https?://'),
    'uuid':     re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I),
}

def check_format_encoding(df: pd.DataFrame, schema: Optional[SchemaOutput] = None) -> List[ValidationCheck]:
    checks = []

    # Whitespace and casing consistency
    for col in df.select_dtypes(include='object').columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        # Whitespace check
        has_leading_trailing = int((series != series.str.strip()).sum())
        if has_leading_trailing > 0:
            checks.append(_check("format_encoding", f"'{col}' — no leading/trailing whitespace",
                passed=False, violations=has_leading_trailing,
                details=f"{has_leading_trailing} values have leading/trailing spaces"))

        # Mojibake/garbled unicode: presence of replacement character or unusual control chars
        mojibake = int(series.str.contains('\ufffd|\x00|\x01|\x02', regex=True, na=False).sum())
        if mojibake > 0:
            checks.append(_check("format_encoding", f"'{col}' — no garbled unicode (mojibake)",
                passed=False, violations=mojibake,
                details=f"{mojibake} values contain replacement characters or control chars"))

        # Sentinel value detection
        coerced = pd.to_numeric(series, errors='coerce')
        parsed_count = coerced.notna().sum()
        total_count = len(series)
        if parsed_count > 0.5 * total_count and parsed_count < total_count:
            sentinels = series[coerced.isna()].value_counts()
            sentinel_count = total_count - parsed_count
            sentinel_examples = ", ".join([f"'{k}'" for k in sentinels.head(3).index])
            checks.append(_check("format_encoding", f"'{col}' contains sentinel strings preventing numeric casting",
                passed=False, violations=sentinel_count,
                details=f"{sentinel_count} non-numeric string values found (e.g. {sentinel_examples}) in otherwise numeric column"))

    # Format pattern conformity based on column name hints
    inferred_types = schema.inferred_types if schema else {}
    for col in df.columns:
        domain = inferred_types.get(col, '').lower()
        if not domain:
            # Fallback: guess from column name
            col_lower = col.lower()
            for key in FORMAT_PATTERNS:
                if key in col_lower:
                    domain = key
                    break
        if domain in FORMAT_PATTERNS:
            pattern = FORMAT_PATTERNS[domain]
            series = df[col].dropna().astype(str)
            mismatches = int((~series.str.match(pattern)).sum())
            checks.append(_check("format_encoding", f"'{col}' matches {domain} pattern",
                passed=(mismatches == 0), violations=mismatches,
                details=f"{mismatches} values do not match expected {domain} format"))

    return checks


# ── Category 7: Regression (re-profile cleaned output) ───────────────────────

def check_regression(
    df_before: pd.DataFrame,
    df_after:  pd.DataFrame,
    profiler_before: Optional[dict] = None
) -> Tuple[List[ValidationCheck], Dict[str, int]]:
    checks = []
    new_nulls: Dict[str, int] = {}
    n_before = len(df_before)
    n_after  = len(df_after)

    # Row count check
    row_drop = n_before - n_after
    row_drop_pct = _pct(row_drop, n_before) * 100
    checks.append(_check("regression", "Row count change is within expectations",
        passed=(row_drop_pct <= 5.0),
        violations=row_drop if row_drop > 0 else 0,
        details=f"Rows: {n_before:,} → {n_after:,} ({row_drop:+,} rows, {round(row_drop_pct, 2)}% change)"))

    # Column count check
    col_before = len(df_before.columns)
    col_after  = len(df_after.columns)
    checks.append(_check("regression", "Column count matches expectations",
        passed=(col_after >= col_before - 2),  # Allow up to 2 intentional drops
        violations=max(0, col_before - col_after),
        details=f"Columns: {col_before} → {col_after}"))

    # New nulls introduced by the cleaning code itself
    for col in df_before.columns:
        if col in df_after.columns:
            null_before = int(df_before[col].isnull().sum())
            null_after  = int(df_after[col].isnull().sum())
            if null_after > null_before:
                introduced = null_after - null_before
                new_nulls[col] = introduced
                checks.append(_check("regression", f"'{col}' — cleaning code did not introduce new nulls",
                    passed=False, violations=introduced,
                    details=f"{introduced} new null values introduced by cleaning (null before: {null_before}, after: {null_after})"))

    if not new_nulls:
        checks.append(_check("regression", "No new nulls introduced by cleaning code",
            passed=True, details="Cleaning code did not introduce any new null values"))

    return checks, new_nulls


# ── Escalation logic ─────────────────────────────────────────────────────────

def compute_escalations(report: ValidationReport) -> List[EscalationFlag]:
    flags = []

    # High-imputation columns
    for diff in report.stats_diff:
        if diff.imputation_pct > 0.20:
            flags.append(EscalationFlag(
                column=diff.column,
                reason=f"{round(diff.imputation_pct*100, 1)}% of values are now synthetically imputed — downstream analysis should note this column's statistics are partially artificial",
                severity="high"
            ))
        elif diff.imputation_pct > 0.10:
            flags.append(EscalationFlag(
                column=diff.column,
                reason=f"{round(diff.imputation_pct*100, 1)}% imputation rate — moderate artificial inflation",
                severity="medium"
            ))

    # Business rule violations
    for v in report.business_rule_violations:
        if v.violation_pct > 0.05:
            flags.append(EscalationFlag(
                column=None,
                reason=f"Business rule '{v.rule}' has {v.violation_count} violations ({round(v.violation_pct*100, 1)}%) — disposition: {v.disposition}",
                severity="high"
            ))

    # New nulls
    for col, count in report.new_nulls_introduced.items():
        flags.append(EscalationFlag(
            column=col,
            reason=f"Cleaning code introduced {count} new null values in '{col}' — possible bug in generated code",
            severity="high"
        ))

    # Structural missingness patterns that should not be imputed
    for p in report.missingness_patterns:
        if p.pattern_type == "structural" and p.co_null_pct > 0.9:
            flags.append(EscalationFlag(
                column=", ".join(p.columns[:3]),
                reason=p.description,
                severity="medium"
            ))

    return flags


# ── Main public interface ─────────────────────────────────────────────────────

@traceable
def run_quality_check(csv_path: str, previous_anomalies: Optional[List[Dict[str, Any]]] = None) -> QualityOutput:
    """
    Backward-compatible lightweight quality check (used for Phase 1 pre-clean scan).
    Returns a simple QualityOutput with basic anomaly list.
    """
    df = load_dataframe(csv_path)
    anomalies: List[QualityAnomaly] = []

    num_rows = len(df)
    if num_rows == 0:
        return QualityOutput(
            is_valid=False,
            anomalies=[QualityAnomaly(anomaly_type="empty_file", description="The dataset has 0 rows.", severity="high")],
            summary="Dataset is entirely empty."
        )

    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        anomalies.append(QualityAnomaly(
            anomaly_type="duplicate_rows",
            description=f"Found {duplicate_count} exact duplicate rows.",
            severity="medium",
            affected_count=int(duplicate_count),
            affected_percentage=float(duplicate_count) / num_rows
        ))

    # Missing values per column
    missing_counts = df.isnull().sum()
    for col_name, count in missing_counts.items():
        if count > 0:
            percentage = float(count) / num_rows
            severity = "high" if percentage > 0.5 else ("medium" if percentage > 0.05 else "low")
            anomalies.append(QualityAnomaly(
                column=col_name,
                anomaly_type="missing_values",
                description=f"Column '{col_name}' has {count} missing values.",
                severity=severity,
                affected_count=int(count),
                affected_percentage=percentage
            ))

    # Mixed data types
    for col_name in df.select_dtypes(include=['object']).columns:
        types = df[col_name].dropna().apply(type).unique()
        if len(types) > 1:
            type_names = [t.__name__ for t in types]
            anomalies.append(QualityAnomaly(
                column=col_name,
                anomaly_type="mixed_types",
                description=f"Column '{col_name}' contains mixed data types: {type_names}.",
                severity="medium"
            ))

    # 7-Layer Context: Expose deeper violations to the Planner
    # 1. Range bounds
    for check in check_range_bounds(df):
        if not check.passed:
            anomalies.append(QualityAnomaly(
                anomaly_type="range_bound_violation",
                description=f"{check.name}. Details: {check.details}",
                severity="high" if check.violations > num_rows * 0.05 else "medium",
                affected_count=check.violations,
                affected_percentage=float(check.violations) / num_rows if num_rows > 0 else 0
            ))
            
    # 2. Cross-field & Business Rules
    cf_checks, _, _ = check_cross_field(df, None)
    for check in cf_checks:
        if not check.passed:
            anomalies.append(QualityAnomaly(
                anomaly_type="cross_field_violation",
                description=f"{check.name}. Details: {check.details}",
                severity="high" if check.violations > num_rows * 0.05 else "medium",
                affected_count=check.violations,
                affected_percentage=float(check.violations) / num_rows if num_rows > 0 else 0
            ))
            
    # 3. Format & Whitespace
    for check in check_format_encoding(df, None):
        if not check.passed:
            anomalies.append(QualityAnomaly(
                anomaly_type="format_violation",
                description=f"{check.name}. Details: {check.details}",
                severity="medium" if check.violations > num_rows * 0.05 else "low",
                affected_count=check.violations,
                affected_percentage=float(check.violations) / num_rows if num_rows > 0 else 0
            ))

    is_valid = len([a for a in anomalies if a.severity in ["high"]]) == 0
    summary = f"Detected {len(anomalies)} anomalies. "
    summary += "The dataset is generally of good quality." if is_valid else "The dataset requires significant cleaning due to high-severity issues."

    return QualityOutput(is_valid=is_valid, anomalies=anomalies, summary=summary)


@traceable
def run_full_validation(
    input_path: str,
    output_path: str,
    schema: Optional[SchemaOutput] = None,
    profiler_before: Optional[dict] = None,
    plan_actions: Optional[List[dict]] = None
) -> ValidationReport:
    """
    7-category full validation. Compares the cleaned output against the original input.
    This is called in Phase 4 (node_validate) after execution.
    """
    df_before = load_dataframe(input_path)
    df_after  = load_dataframe(output_path)

    report = ValidationReport(
        rows_before=len(df_before),
        rows_after=len(df_after),
        cols_before=len(df_before.columns),
        cols_after=len(df_after.columns),
    )

    # 1. Schema
    schema_checks, cols_before, cols_after, dropped = check_schema(df_before, df_after, schema)
    report.schema_checks   = schema_checks
    report.columns_before  = cols_before
    report.columns_after   = cols_after
    report.columns_dropped = dropped
    report.columns_added   = [c for c in cols_after if c not in cols_before]

    # 2. Range & bounds (on cleaned output)
    report.range_checks = check_range_bounds(df_after)

    # 3. Cross-field (on cleaned output)
    cf_checks, biz_violations, miss_patterns = check_cross_field(df_after, schema)
    report.cross_field_checks       = cf_checks
    report.business_rule_violations = biz_violations
    report.missingness_patterns     = miss_patterns

    # 4. Distributional
    dist_checks, stats_diffs = check_distributional(df_before, df_after, plan_actions)
    report.distributional_checks = dist_checks
    report.stats_diff            = stats_diffs

    # 5. Duplicates (on cleaned output)
    dup_checks, key_dup_total = check_duplicates(df_after, schema)
    report.duplicate_checks       = dup_checks
    report.key_based_duplicate_count = key_dup_total

    # 6. Format / encoding (on cleaned output)
    report.format_checks = check_format_encoding(df_after, schema)

    # 7. Regression
    reg_checks, new_nulls = check_regression(df_before, df_after, profiler_before)
    report.regression_checks    = reg_checks
    report.new_nulls_introduced = new_nulls

    # Escalations
    report.escalation_flags = compute_escalations(report)

    # Totals
    all_checks = (
        report.schema_checks + report.range_checks + report.cross_field_checks +
        report.distributional_checks + report.duplicate_checks +
        report.format_checks + report.regression_checks
    )
    report.total_checks  = len(all_checks)
    report.total_passed  = sum(1 for c in all_checks if c.passed)
    report.total_failed  = sum(1 for c in all_checks if not c.passed)
    report.overall_passed = report.total_failed == 0

    return report
