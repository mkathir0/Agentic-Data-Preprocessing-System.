from typing import List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./data_engineer.db"
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    GROQ_API_KEY: str = ""
    API_KEY: str = "default_secret_key"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

    # Observability & LangSmith Tracing
    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "agentic-data-preprocessing-pipeline"
    LANGSMITH_AGENT_ID: str = "agentic-data-preprocessing-pipeline"
    LANGSMITH_AGENT_ENVIRONMENT: str = "production"
    LANGSMITH_API_KEY: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def get_settings() -> Settings:
    load_dotenv(override=True)
    return Settings()

settings = get_settings()
