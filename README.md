<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-1C1C1C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</div>

<h1 align="center">Agentic Data Preprocessing System</h1>

<p align="center">
  <i>Multi-agent system that autonomously profiles, cleans, validates, and documents raw datasets using LLM-driven decision-making over deterministic data engineering tools.</i>
</p>

---

## 🚀 Features

- **Autonomous Code Generation**: Uses `LangChain` and `Groq` (Llama-3.3-70b-versatile) to write pandas cleaning scripts tailored exactly to your dataset's anomalies.
- **LangGraph Orchestration**: A robust AI state machine that enforces a cyclic feedback loop (`Profile -> Plan -> Code -> Execute -> Validate`).
- **Self-Healing Circuit Breaker**: If the generated code fails validation, the LLM catches its own mistake, reads the error stack trace, and rewrites the code (up to 3 retries).
- **Production-Grade Security**: Protected by strict CORS `ALLOWED_ORIGINS`, `X-API-Key` authentication, and HTTP security headers.
- **Rich Markdown Reports**: Automatically outputs detailed engineering reports outlining the transformation plan and the data quality improvements.

---

## 🧠 Architecture

The system operates as a **State Machine** built with `LangGraph`. When a user uploads a CSV, the backend spawns a background task that traverses the following graph nodes:

```mermaid
graph TD
    A([User Uploads CSV]) -->|Trigger Background Task| B[Profile & Plan Node]
    
    subgraph AI Data Engineer Loop
        B -->|Pandera + Schema Inference| C[Generate & Execute Node]
        C -->|Write Python + Run Sandbox| D[Validate Node]
        
        D -->|High Severity Anomalies Found?| E{Circuit Breaker}
        E -->|Yes (Retry < 3)| C
    end
    
    E -->|No / Max Retries| F[Report Node]
    F -->|Generate Markdown Report| G([Job Completed])
    
    style B fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    style C fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff
    style D fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    style E fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff
    style F fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:#fff
```

### The Nodes:
1. **Profile & Plan**: Runs deterministic Pandas profiling (missing values, schema types). Prompts the LLM to create a structured `TransformationPlan`.
2. **Generate & Execute**: The LLM writes raw Python pandas code to satisfy the plan. The system executes this code securely against the CSV.
3. **Validate**: A deterministic validator checks the output CSV. If it finds severe anomalies (e.g. remaining NaNs in critical columns), it routes the error string back to the Generator.
4. **Report**: Compiles the execution path, code, and final status into an `engineering_report.md`.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.11+
- `uv` Package Manager

### 2. Environment Variables
Create a `.env` file in the root directory:
```ini
# LLM Configuration
LLM_MODEL="groq/llama-3.3-70b-versatile"
GROQ_API_KEY="your_groq_api_key_here"

# Security Configuration
API_KEY="your_secret_api_key"
ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"
```

### 3. Install & Run
Run the FastAPI application locally:
```bash
# Sync dependencies
uv sync

# Run the Uvicorn server
uv run uvicorn app.main:app --reload
```

---

## 📡 API Endpoints

All endpoints require the `X-API-Key: <your_secret_api_key>` header.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/jobs/upload` | Upload a messy `.csv` to trigger the AI loop. |
| `GET`  | `/api/jobs/`       | List all historical pipeline jobs. |
| `GET`  | `/api/jobs/{id}`   | Poll for realtime job progress and status. |
| `GET`  | `/api/jobs/{id}/download` | Download the finalized, cleaned CSV. |
| `GET`  | `/api/jobs/{id}/report`   | Download the generated Markdown engineering report. |

---

<p align="center">
  <i>Built securely for automated agentic orchestration.</i>
</p>
