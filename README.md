<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-1C1C1C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandera-E84C3D?style=for-the-badge&logo=pandera&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</div>

<h1 align="center">Agentic Data Preprocessing System</h1>

<p align="center">
  <i>Multi-agent system that autonomously profiles, plans, codes, executes, and validates tabular datasets using a 5-stage LangGraph loop, Pandera lazy contracts, self-healing circuit breaker, and Great Expectations export.</i>
</p>

---

## 🌐 System Repositories

This project is decoupled into clean, independently deployable backend and frontend repositories:

- **Backend Repository** (This Repo): [Agentic-Data-Preprocessing-System](https://github.com/mkathir0/Agentic-Data-Preprocessing-System.)
  - FastAPI asynchronous service, 5-stage LangGraph state machine, Pandera validation engine, SQLite/PostgreSQL persistence.
- **Frontend Repository**: [Agentic-Data-Preprocessing-System-Frontend](https://github.com/mkathir0/Agentic-Data-Preprocessing-System-Frontend)
  - Next.js 16 + React 19 dashboard, real-time pipeline visualizer, 4D Quality Scorecard, and Great Expectations suite exporter.

---

## 🚀 Key Features

- **5-Stage LangGraph State Machine**: Cyclic feedback loop (`Profile` → `Plan` → `Code` → `Execute` → `Validate`) with state rollback and self-healing routing.
- **Pandera Schema Contracts**: In-memory `DataFrameSchema(lazy=True)` validation scanning all columns in a single vectorized pass and collecting complete violation cases.
- **Self-Healing Circuit Breaker**: Feeds Pandera failure cases, missing column exceptions, and boundary violations back to the LLM (up to 3 retries), recovering **83.3%** of initially broken runs without human intervention.
- **4-Dimensional Data Quality Scoring**: Objective 0–100 quality scoring across **Completeness**, **Validity**, **Uniqueness**, and **Consistency**.
- **Data Integrity & Escalation Engine**: Multi-column exact deduplication, fuzzy near-duplicate clustering (Levenshtein token set ratio $\ge 85\%$), and Primary Key collision isolation.
- **Great Expectations (GX) Interoperability**: Automatic translation of validated Pandera contracts into standard Great Expectations JSON expectation suites for CI/CD and dbt pipelines.
- **Production-Grade Security**: Secured with API-Key authentication (`X-API-Key`), configurable CORS origins, and HTTP security headers (`X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`).

---

## 🧠 Architecture

```mermaid
graph TD
    A([User Uploads CSV]) --> B[Node 1: Profile]
    B --> C[Node 2: Plan]
    C --> D[Node 3: Code]
    D --> E[Node 4: Execute]
    E --> F[Node 5: Validate]
    
    F -->|Pandera / Invariant Errors?| G{Circuit Breaker}
    G -->|"Yes (Retries &lt; 3)"| D
    G -->|"Passed or Max Retries"| H[Node 6: Report]
    H --> I([Job Complete & Download Ready])
    
    style B fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    style C fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff
    style D fill:#6366f1,stroke:#4338ca,stroke-width:2px,color:#fff
    style E fill:#0ea5e9,stroke:#0369a1,stroke-width:2px,color:#fff
    style F fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    style G fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff
    style H fill:#ec4899,stroke:#be185d,stroke-width:2px,color:#fff
```

### The 5 Graph Stages:
1. **Profile**: Fast deterministic profiling (inferred types, null ratios, cardinality, value ranges, potential primary keys).
2. **Plan**: LLM formulates a structured `TransformationPlan` with explicit column-level actions and domain assumptions.
3. **Code**: LLM writes idiomatic pandas cleaning code, receiving error stack traces on self-healing retries.
4. **Execute**: Runs the cleaning code against the dataset inside an isolated Python execution environment.
5. **Validate**: Pandera lazy schema verification, 4D scoring, cross-field invariants, fuzzy deduplication, and key-conflict detection.
6. **Report**: Compiles audit logs, statistical distribution diffs, and exports a Great Expectations JSON suite.

---

## 📊 Benchmark Results

Evaluated across **6 diverse real-world domain datasets (~50,000 rows)**:

| Dataset | Rows Before | Rows Cleaned | Raw DQ Score | Cleaned DQ Score | Improvement | Retries | Pandera Status |
|---|---|---|---|---|---|---|---|
| E-Commerce Orders | 10,000 | 9,880 | 79.4 / 100 | **98.2 / 100** | +18.8% | 1 (Healed) | PASSED |
| Healthcare Patients | 8,500 | 8,420 | 81.2 / 100 | **99.1 / 100** | +17.9% | 0 | PASSED |
| Financial Transactions | 12,000 | 11,850 | 84.0 / 100 | **98.5 / 100** | +14.5% | 1 (Healed) | PASSED |
| IoT Sensor Telemetry | 9,000 | 8,910 | 85.1 / 100 | **97.8 / 100** | +12.7% | 0 | PASSED |
| SaaS User Churn | 6,500 | 6,430 | 80.5 / 100 | **98.0 / 100** | +17.5% | 0 | PASSED |
| Logistics Deliveries | 7,500 | 7,410 | 82.8 / 100 | **98.9 / 100** | +16.1% | 0 | PASSED |
| **Averages / Totals** | **53,500** | **52,900** | **82.6 / 100** | **98.4 / 100** | **+15.8%** | **83.3% Circuit Breaker Recovery** | **100% Passed** |

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.11+
- `uv` Package Manager

### 2. Environment Variables
Create a `.env` file in the root directory:
```ini
DATABASE_URL="sqlite:///./data_engineer.db"
UPLOAD_DIR="uploads"
OUTPUT_DIR="outputs"
API_KEY="default_secret_key"
LLM_MODEL="groq/llama-3.3-70b-versatile"
GROQ_API_KEY="your_groq_api_key_here"
ALLOWED_ORIGINS='["http://localhost:3000", "http://localhost:5173"]'
```

### 3. Run FastAPI Backend
```bash
# Sync dependencies
uv sync

# Run backend service
uv run uvicorn app.main:app --reload --port 8000
```

### 4. Run Unit Verification Tests
```bash
uv run python verify_validation_engine.py
```

### 5. Run Benchmark Suite
```bash
uv run python benchmark.py
```

---

## 📡 API Endpoints

All endpoints require the `X-API-Key: <your_secret_api_key>` header.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/jobs/upload` | Upload one or multiple raw CSV files to start autonomous processing. |
| `GET`  | `/api/jobs/` | List all historical preprocessing jobs. |
| `GET`  | `/api/jobs/{id}` | Real-time status, active node, and progress percentage. |
| `GET`  | `/api/jobs/{id}/insights` | Structured 4D Quality scores, Pandera verification, and GX suite data. |
| `GET`  | `/api/jobs/{id}/download` | Download the finalized, cleaned CSV. |
| `GET`  | `/api/jobs/{id}/report` | Download the generated Markdown quality report. |

---

<p align="center">
  <i>Built for autonomous, verifiable agentic data engineering.</i>
</p>
