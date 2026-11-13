import os
from dotenv import load_dotenv

# Load .env FIRST so LangSmith SDK reads the vars at import time
load_dotenv()

# LangSmith Tracing Environment Configuration
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGSMITH_PROJECT", "agentic-data-preprocessing-pipeline")
os.environ.setdefault("LANGCHAIN_PROJECT", "agentic-data-preprocessing-pipeline")
os.environ.setdefault("LANGSMITH_AGENT_ID", "agentic-data-preprocessing-pipeline")
os.environ.setdefault("LANGSMITH_AGENT_ENVIRONMENT", "production")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from app.routers import jobs
from app.config import get_settings

# Create database tables
Base.metadata.create_all(bind=engine)

# Ensure migrations for dynamic columns in sqlite
with engine.connect() as conn:
    from sqlalchemy import text
    try:
        conn.execute(text("ALTER TABLE jobs ADD COLUMN user_instructions TEXT;"))
        conn.commit()
    except Exception:
        pass

app = FastAPI(title="AI Data Engineer Backend")
settings = get_settings()

# Strict CORS for frontend (supports localhost, 127.0.0.1, [::1] on all dev ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

app.include_router(jobs.router)

@app.get("/")
def root():
    return {"message": "Welcome to AI Data Engineer API"}
