import os
import tempfile
import subprocess
import json
from typing import Dict, Any
from langsmith import traceable

@traceable
def execute_cleaning_code(code_str: str, input_csv_path: str, output_csv_path: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Safely executes LLM generated pandas code in an isolated subprocess.
    """
    
    # Wrap the generated code with execution logic
    wrapped_code = f"""
import pandas as pd
import json
import sys
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
        return pd.read_csv(filepath, sep="\\t", **kwargs)
    else:
        return pd.read_csv(filepath, **kwargs)

# --- LLM Generated Code ---
{code_str}
# --------------------------

def main():
    try:
        # 1. Read input dataset
        df = load_dataframe('{input_csv_path}')
        
        # 2. Execute cleaning function
        cleaned_df = clean_data(df)
        
        # 3. Save output dataset
        cleaned_df.to_csv('{output_csv_path}', index=False)
        print(json.dumps({{"status": "success", "message": "Successfully cleaned data"}}))
    except Exception as e:
        print(json.dumps({{"status": "error", "message": str(e)}}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
"""
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
        temp_script.write(wrapped_code)
        temp_script_path = temp_script.name

    result_data = {
        "success": False,
        "logs": "",
        "error": ""
    }

    try:
        # Execute the temporary script using the current Python environment
        # We use sys.executable to ensure it uses the uv venv
        import sys
        process = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        result_data["logs"] = process.stdout
        
        if process.returncode == 0:
            result_data["success"] = True
        else:
            result_data["success"] = False
            result_data["error"] = process.stderr
            
    except subprocess.TimeoutExpired:
        result_data["success"] = False
        result_data["error"] = f"Execution timed out after {timeout} seconds."
    except Exception as e:
        result_data["success"] = False
        result_data["error"] = f"Failed to execute subprocess: {str(e)}"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
            
    return result_data
