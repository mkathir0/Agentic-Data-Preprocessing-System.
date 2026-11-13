"""Read tabular data without silently normalizing identifiers or text tokens."""
import csv
import os
import re
import pandas as pd


def is_identifier_column(name: str) -> bool:
    return bool(re.search(r'(^|_)(id|key|uuid)$|^(id|uuid|key)$', str(name), re.I))


def load_dataframe(filepath: str, **kwargs) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in {'.xlsx', '.xls'}:
        kwargs.setdefault('dtype', object)
        kwargs.setdefault('keep_default_na', False)
        kwargs.setdefault('na_values', [''])
        return pd.read_excel(filepath, **kwargs)
    if ext == '.json':
        kwargs.setdefault('dtype', False)
        kwargs.setdefault('convert_dates', False)
        return pd.read_json(filepath, **kwargs)
    if ext == '.parquet':
        return pd.read_parquet(filepath, **kwargs)
    sep = kwargs.pop('sep', '\t' if ext == '.tsv' else ',')
    # Reject ambiguous headers instead of accepting pandas' automatic renaming.
    with open(filepath, newline='', encoding=kwargs.get('encoding', 'utf-8-sig')) as stream:
        header = next(csv.reader(stream, delimiter=sep), [])
    if len(header) != len(set(header)):
        raise ValueError('Duplicate column names must be resolved before processing.')
    kwargs.setdefault('dtype', object)
    kwargs.setdefault('keep_default_na', False)
    kwargs.setdefault('na_values', [''])
    kwargs.setdefault('skip_blank_lines', False)
    df = pd.read_csv(filepath, sep=sep, **kwargs)
    # Infer only canonical, exactly representable numbers outside ID columns.
    # Values such as 00123, NA, NULL, 10.00 and large precise decimals stay text.
    for col in df.columns:
        values = df[col].dropna()
        if is_identifier_column(col) or values.empty or not values.map(lambda x: isinstance(x, str)).all():
            continue
        if values.map(lambda x: bool(re.fullmatch(r'-?(0|[1-9]\d*)', x))).all():
            try:
                numeric = pd.to_numeric(df[col], errors='raise')
                if values.map(lambda x: str(int(x)) == x and abs(int(x)) <= 2**53).all():
                    df[col] = numeric
            except (ValueError, OverflowError):
                pass
        elif values.map(lambda x: _canonical_float(x)).all():
            df[col] = pd.to_numeric(df[col], errors='raise')
    return df


def _canonical_float(value: str) -> bool:
    try:
        return bool(re.fullmatch(r'-?\d+\.\d+', value)) and str(float(value)) == value
    except (ValueError, OverflowError):
        return False
