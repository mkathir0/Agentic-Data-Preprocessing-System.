import os
import sys
import json
from dotenv import load_dotenv

from app.agents.profiler import run_profiler
from app.agents.quality import run_quality_check
from app.agents.schema_agent import run_schema_inference
from app.agents.planner import generate_transformation_plan
from app.agents.code_generator import generate_cleaning_code

# Load environment variables (e.g. GROQ_API_KEY)
load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python verify_generation.py <path_to_csv>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        sys.exit(1)
        
    print(f"--- 1. Profiling dataset: {csv_file} ---")
    profiler_output = run_profiler(csv_file)
    quality_output = run_quality_check(csv_file)
    schema_output = run_schema_inference(csv_file)
    print("Profiling complete.")
    
    print("\n--- 2. Generating Transformation Plan ---")
    try:
        plan = generate_transformation_plan(profiler_output, quality_output, schema_output)
        print("Generated Plan:")
        print(plan.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error generating plan: {e}")
        return
        
    print("\n--- 3. Generating Python Code ---")
    try:
        code_output = generate_cleaning_code(plan)
        print("Generated Code:")
        print("========================================")
        print(code_output.code)
        print("========================================")
        print(f"Dependencies: {code_output.dependencies}")
    except Exception as e:
        print(f"Error generating code: {e}")

if __name__ == "__main__":
    main()
