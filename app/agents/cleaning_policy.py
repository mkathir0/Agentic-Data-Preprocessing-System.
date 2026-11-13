"""Deterministic cleaning actions shared by the planner, executor and validator.

Only transformations with explicit, reproducible semantics are executable. The
policy normalizes representation; it never imputes values, fixes contradictory
business records, or fuzzy-merges categories.
"""
from collections import Counter
from datetime import datetime
import re

import pandas as pd

# This registry is the automatic-repair contract. Every nontrivial repair must
# be independently proposed by the deterministic profiler on the input frame.
# Keep semantic corrections (email repair, key reconciliation, imputation, etc.)
# out of this registry until there is a separately tested, authoritative rule.
AUTOMATIC_REPAIR_RULES = {
    "strip_whitespace": "Trim boundary whitespace without changing token content.",
    "drop_duplicates": "Remove exact whole-row duplicates only.",
    "normalize_numeric_format": "Parse supported numeric/currency/unit formats; only recognized missing sentinels become null.",
    "normalize_datetime_format": "Normalize only when every nonmissing value parses and slash-date order is unambiguous.",
    "normalize_boolean": "Normalize recognized boolean tokens in explicitly boolean-named columns.",
    "normalize_category_case": "Unify case-only variants in low-cardinality categories; never merge aliases.",
}
SUPPORTED_ACTIONS = frozenset(AUTOMATIC_REPAIR_RULES)
SENTINEL_TOKENS = {"", "n/a", "na", "null", "none", "nan", "unknown", "missing"}
BOOLEAN_VALUES = {
    "true": True, "t": True, "yes": True, "y": True, "1": True,
    "false": False, "f": False, "no": False, "n": False, "0": False,
}
NUMERIC_COLUMN_HINT = re.compile(r"price|amount|subtotal|tax|total|quantity|qty|discount|percent|pct|rating|duration|(?:^|_)(?:count|units?)(?:_|$)", re.I)
SEMANTIC_COLUMN_HINT = re.compile(
    r"email|e-mail|phone|mobile|address|url|website|(^|_)(?:name|description|comment|notes?)(_|$)", re.I
)


def parse_numeric_value(value):
    """Return (number, is_percent, is_missing_sentinel), or (None, False, False)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None, False, True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), False, False
    text = str(value).strip()
    if text.casefold() in SENTINEL_TOKENS:
        return None, False, True
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    text = re.sub(r"^USD\s*", "", text, flags=re.I).replace(",", "")
    text = re.sub(r"^-\$\s*", "-", text)
    text = re.sub(r"^\$\s*", "", text)
    text = re.sub(r"\s+units?$", "", text, flags=re.I)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return None, False, False
    return float(text), percent, False


def _infer_date_order(values) -> str | None:
    month_first = day_first = False
    for value in values:
        match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(value).strip())
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12 and second <= 12:
            day_first = True
        elif second > 12 and first <= 12:
            month_first = True
    if month_first == day_first:
        return None
    return "month_first" if month_first else "day_first"


def parse_datetime_value(value, date_order: str | None = None):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None, True
    text = str(value).strip()
    if text.casefold() in SENTINEL_TOKENS:
        return None, True
    if re.fullmatch(r"\d{10}", text):
        stamp = pd.to_datetime(int(text), unit="s", errors="coerce")
        return (stamp.to_pydatetime(), False) if not pd.isna(stamp) else (None, False)
    formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"]
    if date_order == "month_first":
        formats.append("%m/%d/%Y")
    elif date_order == "day_first":
        formats.append("%d/%m/%Y")
    elif re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
        return None, False
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt), False
        except ValueError:
            continue
    return None, False


def suggest_safe_format_actions(df: pd.DataFrame, categorical_columns: list[str] | None = None) -> list[dict]:
    """Derive format-only actions from values; uncertain columns are not proposed."""
    actions: list[dict] = []
    for col in df.columns:
        values = df[col].dropna()
        if values.empty or re.search(r"(^|_)(id|key|uuid)$", col, re.I):
            continue
        if NUMERIC_COLUMN_HINT.search(col):
            parsed = [parse_numeric_value(value) for value in values]
            if all(number is not None or missing for number, _, missing in parsed):
                decorated = any(percent or re.search(r"\$|\bUSD\b|,|\bunits?\b", str(value), re.I) for value, (_, percent, _) in zip(values, parsed))
                sentinels = any(missing for _, _, missing in parsed)
                if decorated or sentinels:
                    percent_mode = bool(re.search(r"percent|pct|discount", col, re.I))
                    if percent_mode:
                        percent_numbers = {number for number, is_percent, _ in parsed if is_percent}
                        fractional_numbers = {number for number, is_percent, missing in parsed if not is_percent and not missing and 0 < abs(number) < 1}
                        if fractional_numbers and percent_numbers:
                            aligned_scales = all(round(number * 100, 6) in percent_numbers for number in fractional_numbers)
                            if not aligned_scales:
                                continue
                        raw_one = any(number == 1 and not is_percent for number, is_percent, _ in parsed)
                        if raw_one and percent_numbers:
                            continue
                    actions.append({
                        "action_type": "normalize_numeric_format", "column": col,
                        "confidence": "high", "explanation": "Normalize consistently parseable numeric/currency/unit strings; map only recognized sentinels to missing.",
                        "parameters": {"percent_points": percent_mode},
                    })
        if re.search(r"date|time", col, re.I):
            order = _infer_date_order(values)
            parsed_dates = [parse_datetime_value(value, order) for value in values]
            if all(value is not None or missing for value, missing in parsed_dates):
                noncanonical = any(
                    not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?", str(value))
                    for value in values
                )
                ambiguous_slash = any(re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", str(value).strip()) for value in values) and order is None
                if noncanonical and not ambiguous_slash:
                    actions.append({
                        "action_type": "normalize_datetime_format", "column": col,
                        "confidence": "high", "explanation": "Normalize consistently parseable date representations without changing chronology.",
                        "parameters": {"date_order": order},
                    })
        if re.match(r"^(is|has|was|can)_", col, re.I):
            tokens = {str(value).strip().casefold() for value in values}
            if tokens and tokens <= set(BOOLEAN_VALUES) and not tokens <= {"true", "false"}:
                actions.append({
                    "action_type": "normalize_boolean", "column": col, "confidence": "high",
                    "explanation": "Normalize recognized true/false tokens to boolean values.", "parameters": {},
                })
    for col in categorical_columns or []:
        if (col not in df.columns or NUMERIC_COLUMN_HINT.search(col)
                or SEMANTIC_COLUMN_HINT.search(col)
                or re.search(r"(^|_)(id|key|uuid|sku|code)$", col, re.I)
                or re.match(r"^(is|has|was|can)_", col, re.I)):
            continue
        values = df[col].dropna()
        groups: dict[str, set[str]] = {}
        for value in values:
            if isinstance(value, str):
                groups.setdefault(value.strip().casefold(), set()).add(value.strip())
        if any(len(group) > 1 for group in groups.values()):
            actions.append({
                "action_type": "normalize_category_case", "column": col, "confidence": "high",
                "explanation": "Unify values that differ only by letter case, using the most frequent spelling; do not merge aliases or fuzzy matches.",
                "parameters": {},
            })
    return actions


def _candidate_actions_for_frame(df: pd.DataFrame) -> list[dict]:
    """Rebuild the profiler's evidence from the exact frame being cleaned."""
    categorical = []
    for col in df.columns:
        if df[col].dtype == object and len(df):
            nonmissing = max(int(df[col].notna().sum()), 1)
            if 0 < df[col].nunique(dropna=True) / nonmissing < 0.1:
                categorical.append(col)
    return suggest_safe_format_actions(df, categorical)


def _matches_evidenced_candidate(action: dict, candidates: list[dict]) -> bool:
    kind, column = action.get("action_type"), action.get("column")
    params = action.get("parameters") or {}
    for candidate in candidates:
        if (candidate.get("action_type"), candidate.get("column")) != (kind, column):
            continue
        expected = candidate.get("parameters") or {}
        # Defaults are explicit in the deterministic rule, but older planner
        # outputs may omit them. Compare the effective values consistently.
        effective = {**expected, **params}
        if effective == expected:
            return True
    return False


def _apply_action(result: pd.DataFrame, action: dict) -> tuple[pd.DataFrame, str | None]:
    kind, column = action.get("action_type"), action.get("column")
    params = action.get("parameters") or {}
    if column is not None and column not in result.columns:
        return result, f"Unknown column in action: {column}"
    if kind in {"strip_whitespace", "drop_duplicates"}:
        return result, None
    if column is None:
        return result, f"{kind} requires a specific column; values preserved."
    source = result[column]
    if kind == "normalize_numeric_format":
        parsed = [parse_numeric_value(value) for value in source]
        if any(number is None and not missing for number, _, missing in parsed):
            return result, f"'{column}' has nonnumeric values outside the supported parser; column preserved for review."
        has_percent = any(is_percent for _, is_percent, _ in parsed)
        converted = []
        for number, is_percent, missing in parsed:
            if missing:
                converted.append(float("nan"))
            elif params.get("percent_points") and has_percent and not is_percent and 0 < abs(number) < 1:
                converted.append(number * 100)
            else:
                converted.append(number)
        result[column] = converted
    elif kind == "normalize_datetime_format":
        date_order = params.get("date_order")
        parsed = [parse_datetime_value(value, date_order) for value in source]
        if any(value is None and not missing for value, missing in parsed):
            return result, f"'{column}' contains an unsupported or ambiguous date; column preserved for review."
        result[column] = [
            pd.NA if missing else value.strftime("%Y-%m-%dT%H:%M:%S")
            for value, missing in parsed
        ]
    elif kind == "normalize_boolean":
        tokens = [None if pd.isna(value) else str(value).strip().casefold() for value in source]
        if any(token is not None and token not in BOOLEAN_VALUES for token in tokens):
            return result, f"'{column}' contains unrecognized boolean values; column preserved for review."
        result[column] = [pd.NA if token is None else str(BOOLEAN_VALUES[token]).lower() for token in tokens]
    elif kind == "normalize_category_case":
        values = source.dropna()
        groups: dict[str, Counter] = {}
        for value in values:
            if isinstance(value, str):
                groups.setdefault(value.strip().casefold(), Counter())[value.strip()] += 1
        mapping = {value: counts.most_common(1)[0][0] for counts in groups.values() for value in counts}
        result[column] = source.map(lambda value: mapping.get(value.strip(), value) if isinstance(value, str) else value)
    return result, None


def apply_cleaning_plan(df: pd.DataFrame, actions: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy(deep=True)
    review: list[str] = []
    candidates = _candidate_actions_for_frame(df)
    protected: set[str] = set()
    for action in actions:
        if action.get("action_type") == "flag_for_review" or action.get("confidence") != "high":
            if action.get("column") in df.columns:
                protected.add(action["column"])
    for action in actions:
        kind, column = action.get("action_type"), action.get("column")
        if action.get("confidence") != "high" or kind not in SUPPORTED_ACTIONS:
            review.append(f"{column or 'Dataset'}: {kind} requires a reviewed domain rule; original values preserved.")
            continue
        if column is not None and column not in result.columns:
            review.append(f"Unknown column in action: {column}")
            continue
        if kind == "strip_whitespace":
            columns = [column] if column else list(result.columns)
            for col in columns:
                if col not in protected:
                    result[col] = result[col].map(lambda value: value.strip() if isinstance(value, str) else value)
        elif kind == "drop_duplicates":
            if column is not None:
                review.append("Subset deduplication is not allowed: conflicting records must be preserved.")
            else:
                result = result.drop_duplicates(keep="first")
        elif column in protected:
            review.append(f"'{column}' is protected for review; values preserved.")
        else:
            if kind not in {"strip_whitespace", "drop_duplicates"} and not _matches_evidenced_candidate(action, candidates):
                review.append(
                    f"'{column}' {kind} has no matching deterministic evidence; values preserved for review."
                )
                continue
            result, problem = _apply_action(result, action)
            if problem:
                review.append(problem)
    return result.reset_index(drop=True), review


def frames_match(expected: pd.DataFrame, actual: pd.DataFrame) -> bool:
    """Compare complete records and multiplicities, allowing harmless row order."""
    if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
        return False

    def records(frame):
        normalized = frame.astype("string").replace("", pd.NA)
        return pd.util.hash_pandas_object(normalized, index=False).value_counts().sort_index()

    return records(expected).equals(records(actual))
