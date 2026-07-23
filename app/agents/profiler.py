import pandas as pd
from ydata_profiling import ProfileReport
import json
import io

from app.agents.schemas import ProfilerOutput

def run_profiler(csv_path: str) -> ProfilerOutput:
    """
    Run the profiler agent on a raw CSV file.
    Returns a structured ProfilerOutput Pydantic model.
    """
    df = pd.read_csv(csv_path)
    
    # We use minimal mode for ydata-profiling to be fast and deterministic
    profile = ProfileReport(df, minimal=True)
    description = profile.get_description()
    
    # Extract overall metrics
    table_stats = description.table
    num_rows = table_stats.get('n', 0)
    num_columns = table_stats.get('n_var', 0)
    memory_usage = table_stats.get('memory_size', 0)
    missing_cells = table_stats.get('n_cells_missing', 0)
    missing_percentage = table_stats.get('p_cells_missing', 0.0)
    duplicate_rows = table_stats.get('n_duplicates', 0)
    duplicate_percentage = table_stats.get('p_duplicates', 0.0)
    
    # Extract column metrics
    variables = description.variables
    column_types = {}
    columns_summary = {}
    
    for col_name, col_stats in variables.items():
        # ydata-profiling uses a type class, we get the name of it
        col_type = col_stats['type'].value if hasattr(col_stats['type'], 'value') else str(col_stats['type'])
        column_types[col_name] = col_type
        
        # Clean up stats to be JSON serializable (remove arrays/series if any)
        clean_stats = {}
        for k, v in col_stats.items():
            if isinstance(v, (int, float, str, bool)):
                clean_stats[k] = v
        columns_summary[col_name] = clean_stats

    return ProfilerOutput(
        num_rows=num_rows,
        num_columns=num_columns,
        memory_usage_bytes=memory_usage,
        missing_cells=missing_cells,
        missing_cells_percentage=missing_percentage,
        duplicate_rows=duplicate_rows,
        duplicate_rows_percentage=duplicate_percentage,
        column_types=column_types,
        columns_summary=columns_summary
    )
