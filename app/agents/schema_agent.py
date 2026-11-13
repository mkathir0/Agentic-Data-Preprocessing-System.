import re
import pandas as pd
from typing import Dict
from langsmith import traceable

from app.agents.schemas import SchemaOutput
from app.utils.data_loader import load_dataframe, is_identifier_column

@traceable(name="Schema Agent", run_type="chain")
def run_schema_inference(csv_path: str) -> SchemaOutput:
    """
    Analyzes the dataframe to infer primary keys, foreign keys, and semantic types.
    Returns a SchemaOutput model.
    """
    df = load_dataframe(csv_path)
    num_rows = len(df)
    
    primary_keys = []
    candidate_fks = []
    categorical_columns = []
    datetime_columns = []
    inferred_types: Dict[str, str] = {}

    if num_rows == 0:
        return SchemaOutput(
            primary_keys=[],
            candidate_foreign_keys=[],
            categorical_columns=[],
            datetime_columns=[],
            inferred_types={}
        )

    for col in df.columns:
        series = df[col]
        non_null_series = series.dropna()
        num_non_null = len(non_null_series)
        num_unique = non_null_series.nunique()

        # 1. Primary Key Inference (High cardinality, unique, typically ID or hash)
        if is_identifier_column(col) and num_non_null == num_rows and num_unique / max(num_rows, 1) >= 0.90:
            primary_keys.append(col)
            inferred_types[col] = "Identifier"
            continue # Don't classify PKs as categorical or FKs easily

        # 2. Foreign Key Inference (Has ID in name, but not unique)
        if is_identifier_column(col) and num_unique < num_non_null:
            candidate_fks.append(col)
            inferred_types[col] = "Foreign Key"

        # 3. Datetime Inference
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(col)
            inferred_types[col] = "Datetime"
        elif (series.dtype == 'object' or pd.api.types.is_string_dtype(series)) and re.search(r'date|time|_at$', col, re.I) and num_non_null > 0:
            # Attempt a quick parse to see if it's mostly datetime
            try:
                # Sample a few rows to speed up, or just try on non-null
                pd.to_datetime(non_null_series.head(100), errors='raise')
                datetime_columns.append(col)
                inferred_types[col] = "Datetime"
            except (ValueError, TypeError, pd.errors.ParserError):
                pass
                
        # 4. Categorical Columns (Low cardinality strings)
        if col not in datetime_columns and col not in primary_keys:
            if series.dtype == 'object' or series.dtype.name == 'category':
                if num_unique > 0 and num_unique / num_non_null < 0.1: # Less than 10% unique
                    categorical_columns.append(col)
                    inferred_types[col] = "Categorical"
                else:
                    inferred_types[col] = "Text"
            elif pd.api.types.is_numeric_dtype(series):
                inferred_types[col] = "Numeric"
            elif pd.api.types.is_bool_dtype(series):
                inferred_types[col] = "Boolean"

    return SchemaOutput(
        primary_keys=primary_keys,
        candidate_foreign_keys=candidate_fks,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        inferred_types=inferred_types
    )
