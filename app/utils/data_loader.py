import pandas as pd
import os

def load_dataframe(filepath: str, **kwargs) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath, **kwargs)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath, **kwargs)
    elif ext == ".json":
        return pd.read_json(filepath, **kwargs)
    elif ext == ".parquet":
        return pd.read_parquet(filepath, **kwargs)
    elif ext == ".tsv":
        return pd.read_csv(filepath, sep="\t", **kwargs)
    else:
        # Fallback to csv
        return pd.read_csv(filepath, **kwargs)
