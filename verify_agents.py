import os
import sys
import json

from app.agents.profiler import run_profiler
from app.agents.quality import run_quality_check
from app.agents.schema_agent import run_schema_inference

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python verify_agents.py <path_to_csv>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        sys.exit(1)
        
    print(f"Testing agents with dataset: {csv_file}")
    
    try:
        print("\n--- Running Profiler Agent ---")
        profiler_output = run_profiler(csv_file)
        print(profiler_output.model_dump_json(indent=2))
        
        print("\n--- Running Quality Agent ---")
        quality_output = run_quality_check(csv_file)
        print(quality_output.model_dump_json(indent=2))
        
        print("\n--- Running Schema Agent ---")
        schema_output = run_schema_inference(csv_file)
        print(schema_output.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"Error during agent execution: {e}")

if __name__ == "__main__":
    main()
