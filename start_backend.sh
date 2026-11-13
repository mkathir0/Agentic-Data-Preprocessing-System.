#!/usr/bin/env bash
# ==============================================================================
# Agentic Data Preprocessing System - Backend Startup Script
# ==============================================================================

set -e

# Resolve backend directory regardless of where the script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🚀 Starting Agentic Data Preprocessing Backend..."
echo "============================================================"

# 1. Ensure required runtime directories exist
mkdir -p uploads outputs

# 2. Check environment configuration (.env)
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    echo "⚠️  No .env file found. Copying defaults from .env.example..."
    cp .env.example .env
  else
    cat <<EOF > .env
DATABASE_URL=sqlite:///./data_engineer.db
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
API_KEY=default_secret_key
LLM_MODEL=groq/llama-3.3-70b-versatile
GROQ_API_KEY=
ALLOWED_ORIGINS='["http://localhost:3000", "http://localhost:5173"]'
EOF
    echo "✓ Created new .env file with default settings."
  fi
fi

# 3. Check GROQ_API_KEY status
if grep -q "^GROQ_API_KEY=$" .env 2>/dev/null || grep -q '^GROQ_API_KEY=""$' .env 2>/dev/null || ! grep -q "^GROQ_API_KEY=" .env 2>/dev/null; then
  echo "⚠️  NOTE: GROQ_API_KEY is not set in .env."
  echo "   The server will start, but LLM planning & code generation require a key."
  echo "   Add your key to .env: GROQ_API_KEY=gsk_..."
fi

# 4. Check if port 8000 is occupied
PORT_PID=$(lsof -ti:8000 2>/dev/null || true)
if [ -n "$PORT_PID" ]; then
  echo "⚠️  Port 8000 is currently occupied by PID $PORT_PID."
  echo "   Stopping previous process on port 8000..."
  kill -9 $PORT_PID 2>/dev/null || true
  sleep 1
  echo "✓ Port 8000 is now free."
fi

# 5. Determine runtime command (uv or local virtualenv)
if command -v uv >/dev/null 2>&1; then
  echo "✓ Using uv runtime manager"
  CMD="uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
elif [ -f "$SCRIPT_DIR/.venv/bin/uvicorn" ]; then
  echo "✓ Using local .venv uvicorn"
  CMD="$SCRIPT_DIR/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
elif command -v uvicorn >/dev/null 2>&1; then
  echo "✓ Using system uvicorn"
  CMD="uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
else
  echo "❌ Error: Could not find 'uv' or 'uvicorn'."
  echo "   Please install dependencies with: uv sync"
  exit 1
fi

echo "------------------------------------------------------------"
echo "🌐 API Base URL:       http://localhost:8000/api"
echo "📖 Swagger API Docs:   http://localhost:8000/docs"
echo "🔑 Default API Key:    default_secret_key"
echo "🖥️  Frontend URL:       http://localhost:3000"
echo "------------------------------------------------------------"
echo "Starting FastAPI server with auto-reload enabled (Ctrl+C to stop)..."
echo ""

exec $CMD

