import os
from app.agents.executor import execute_cleaning_code

def main():
    input_csv = "test_data/sample.csv"
    output_csv = "test_data/cleaned_sample.csv"
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} does not exist.")
        return

    print("--- Test 1: Valid Code Execution ---")
    valid_code = """
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Just a simple transformation
    df['new_column'] = 'test'
    return df
"""
    result = execute_cleaning_code(valid_code, input_csv, output_csv)
    print(f"Success: {result['success']}")
    print(f"Logs: {result['logs'].strip()}")
    if not result['success']:
        print(f"Error: {result['error']}")
    else:
        print(f"Cleaned CSV generated at: {output_csv}")

    print("\n--- Test 2: Buggy Code Execution ---")
    buggy_code = """
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Intentionally raising an error
    df['missing_col'] = df['non_existent_column'] + 1
    return df
"""
    result2 = execute_cleaning_code(buggy_code, input_csv, output_csv)
    print(f"Success: {result2['success']}")
    print(f"Error Caught Safely: {result2['error'].strip()}")

if __name__ == "__main__":
    main()
