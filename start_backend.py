#!/usr/bin/env python3
"""
Agentic Data Preprocessing System - Cross-Platform Backend Starter
Usage:
    python start_backend.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent
    os.chdir(root_dir)

    print("=" * 60)
    print("🚀 Starting Agentic Data Preprocessing Backend...")
    print("=" * 60)

    # 1. Ensure runtime directories exist
    (root_dir / "uploads").mkdir(exist_ok=True)
    (root_dir / "outputs").mkdir(exist_ok=True)

    # 2. Check .env
    env_file = root_dir / ".env"
    if not env_file.exists():
        example_env = root_dir / ".env.example"
        if example_env.exists():
            print("⚠️  No .env file found. Copying defaults from .env.example...")
            shutil.copy(example_env, env_file)
        else:
            with open(env_file, "w") as f:
                f.write(
                    "DATABASE_URL=sqlite:///./data_engineer.db\n"
                    "UPLOAD_DIR=uploads\n"
                    "OUTPUT_DIR=outputs\n"
                    "API_KEY=default_secret_key\n"
                    "LLM_MODEL=groq/llama-3.3-70b-versatile\n"
                    "GROQ_API_KEY=\n"
                    "ALLOWED_ORIGINS='[\"http://localhost:3000\", \"http://localhost:5173\"]'\n"
                )
            print("✓ Created new .env file with default settings.")

    # 3. Check for uv
    has_uv = shutil.which("uv") is not None
    if has_uv:
        cmd = ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    else:
        # Fall back to sys.executable with -m uvicorn
        cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

    print("------------------------------------------------------------")
    print("🌐 API Base URL:       http://localhost:8000/api")
    print("📖 Swagger API Docs:   http://localhost:8000/docs")
    print("🔑 Default API Key:    default_secret_key")
    print("🖥️  Frontend URL:       http://localhost:3000")
    print("------------------------------------------------------------")
    print(f"Starting server via: {' '.join(cmd)}")
    print("Press Ctrl+C to stop.\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n✓ Backend stopped.")

if __name__ == "__main__":
    main()

