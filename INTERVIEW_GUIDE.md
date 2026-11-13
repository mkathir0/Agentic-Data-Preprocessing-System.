# AI Data Preprocessing System — Interview Cheat Sheet & Architecture Guide

This guide gives you everything you need to confidently discuss, explain, and defend this project in software engineering and data engineering interviews.

---

## 1. Resume Bullet Points: Deconstructed & Explained

### Bullet 1: 5-Stage LangGraph Loop with Dynamic Pandas Execution
> **Resume Point:** *Built a 5-stage LangGraph loop (Profile → Plan → Code → Execute → Validate) with dynamic pandas execution.*

- **What it means in plain English:**
  Most LLM data cleaning scripts are one-shot: you pass a prompt and hope for the best. In this project, you implemented an autonomous state machine using **LangGraph**. The workflow moves through 5 distinct nodes:
  1. **Profile (`node_profile`):** Calculates descriptive statistics, infers column datatypes, and scans initial anomalies deterministically.
  2. **Plan (`node_plan`):** Feeds only structured JSON metadata (never raw private CSVs) to an LLM to create a structured `TransformationPlan` with step-by-step rationales, confidence scores, and assumptions.
  3. **Code (`node_code`):** Generates standalone Python pandas cleaning code.
  4. **Execute (`node_execute`):** Runs the Python script in an isolated subprocess sandbox with timeout protection, writing out `cleaned.csv`.
  5. **Validate (`node_validate`):** Runs the validation engine against the output.
- **Where it lives in the code:** [`app/agents/graph.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/graph.py), [`app/agents/executor.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/executor.py).
- **Interview Talking Point:**
  > *"Instead of a naive linear prompt chain, I used LangGraph to model the data cleaning lifecycle as a finite state machine. This allows the agent to cycle back if validation fails, capturing state across retries rather than silently failing."*

---

### Bullet 2: Self-Healing Circuit Breaker via Pandera (Saving 30%+ of Runs)
> **Resume Point:** *Added a self-healing circuit breaker feeding Pandera errors to LLM (3 retries), saving 30%+ of runs.*

- **What it means in plain English:**
  When an LLM writes pandas code on its first try, it often makes subtle mistakes: forgetting a boundary check, casting a column to the wrong type, or leaving nulls in primary key columns.
  A **circuit breaker** intercepts these failures before the pipeline finishes:
  1. The validator dynamically synthesizes a **Pandera `DataFrameSchema`** enforcing column contracts (non-null PKs, domain bounds, regex formats, strict dtypes).
  2. We run lazy validation: `schema.validate(df, lazy=True)`.
  3. If checks fail, Pandera collects all failure cases across the entire DataFrame into `err.failure_cases` (column name, check name, failure count, sample failing values).
  4. Instead of crashing, the system injects this structured failure table directly into the LLM's prompt on retry.
  5. On benchmark runs, this self-healing loop recovers **83.3% of failing runs** on the first or second retry!
- **Where it lives in the code:** [`app/agents/validation_engine.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/validation_engine.py) (`validate_pandera_contract`), [`app/agents/graph.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/graph.py) (`route_after_validation`).
- **Interview Talking Point:**
  > *"If you just tell an LLM 'your code failed', it guesses randomly. By using Pandera's lazy validation, I extract the exact failing column, the check that failed, and the offending values. When we pass that contract failure table back to the LLM, it fixes the code with pinpoint precision on the first retry."*

---

### Bullet 3: Job-Based FastAPI Service with Production Security
> **Resume Point:** *Exposed a job-based FastAPI service secured with API-key auth, CORS rules, and HTTP security headers.*

- **What it means in plain English:**
  Data cleaning and LLM generation take time (10–30 seconds). A synchronous endpoint would block the client and risk HTTP gateway timeouts.
  - **Job-based architecture:** The user uploads a CSV (`POST /api/jobs/upload`). The server writes the file, creates a job record in SQLite/Postgres with status `PENDING`, and starts a background worker task. The frontend polls `GET /api/jobs/{id}` for live progress (0% to 100%) and downloads the cleaned CSV and report when finished.
  - **Security:** Requires custom `X-API-Key` authentication header, enforces CORS origin whitelisting (`ALLOWED_ORIGINS`), and injects security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Strict-Transport-Security`).
- **Where it lives in the code:** [`app/main.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/main.py), [`app/security.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/security.py), [`app/routers/jobs.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/routers/jobs.py).

---

### Bullet 4: Cross-Field Validation, Fuzzy Deduplication & Key Conflicts
> **Resume Point:** *Implemented cross-field validation, fuzzy deduplication, and key-conflict detection for data integrity.*

- **What it means in plain English:**
  - **Cross-Field Validation:** Checking business rules that span multiple columns:
    - *Temporal ordering:* `ended_at >= started_at`, `ship_date >= order_date`.
    - *Arithmetic identities:* `subtotal + tax = total_amount`, `monthly_fee * tenure = total_charges`.
  - **Fuzzy Deduplication:** Catches typos and near-duplicate entity strings that exact `df.drop_duplicates()` misses (e.g. `"San Francisco"` vs `"San Fransisco"`, `"Google Inc"` vs `"Google, Inc."`). Implemented via Levenshtein sequence matching and token Jaccard similarity, with smart number-filtering to prevent false alarms on numbered entities like `"Station 1"` vs `"Station 2"`.
  - **Key-Conflict Detection (PK Collisions):** Crucial distinction in data engineering:
    - *Exact duplicate:* Entire row is identical. Safe to drop with `drop_duplicates()`.
    - *Key conflict:* Two rows have the **same Primary Key** (`order_id = 1004`), but **different data** in other columns (e.g. conflicting dollar amounts). Dropping one row silently deletes real business transactions. The engine detects this and flags it for human review.
- **Where it lives in the code:** [`app/agents/validation_engine.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/validation_engine.py) (`detect_fuzzy_duplicates`, `detect_key_conflicts`, `check_cross_field_relations`).

---

### Bullet 5: Escalation Logic Replacing Unreliable LLM Self-Checks
> **Resume Point:** *Designed escalation logic for runs failing validation checks, replacing unreliable LLM self-checks.*

- **What it means in plain English:**
  Naive AI projects ask the LLM: *"Did you clean the data properly? Check your work."* The LLM almost always hallucinates: *"Yes, everything is perfect!"*
  This system replaces LLM self-checks with **ground-truth deterministic Python code**.
  Furthermore, when the system encounters irreconcilable data issues—such as >20% missing values in a critical column, unresolvable primary key collisions, or high imputation that alters statistical distributions—it does **not** allow the LLM to invent synthetic data. It triggers **Escalation Logic**, creating structured `EscalationFlag` entries (`severity: high`, `reason: ...`, `disposition: flagged`) requiring human review.
- **Where it lives in the code:** [`app/agents/validation_engine.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/validation_engine.py) (lines 780–820), [`app/agents/schemas.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/schemas.py) (`EscalationFlag`).

---

### Bullet 6: Multi-Framework Integration (Pandera, LiteLLM/Groq, React)
> **Resume Point:** *Integrated Great Expectations with Pandera; routed LLM queries via LiteLLM/Groq with a React frontend.*

- **What it means in plain English:**
  The project combines the contract enforcement of **Pandera** with Pydantic structured output models, routed through **Groq** (`llama-3.3-70b-versatile`) for sub-second plan and code generation, coupled with a responsive **React** single-page application for file uploads, live status tracking, and report downloads.
- **Where it lives in the code:** [`app/agents/planner.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/planner.py), [`app/agents/code_generator.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/code_generator.py), [`app/config.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/config.py).

---

### Bullet 7: Automated Markdown Reports & 6-Dataset Benchmark
> **Resume Point:** *Auto-generated Markdown quality reports; benchmarked on 5–6 datasets (∼10k rows each).*

- **What it means in plain English:**
  - **Automated Reports:** Every pipeline execution compiles an executive Markdown report containing before/after KPI tables, Pandera verification logs, 7-category validation check breakdowns, human escalation flags, transformation plan audit trails, and the executed Python code.
  - **Benchmark Suite:** Benchmarked across **6 real-world domain datasets (~50,000 total rows)**:
    1. E-Commerce Orders (10,400 rows)
    2. Healthcare Patients (5,150 rows)
    3. Fintech Transactions (10,120 rows)
    4. Bikeshare Mobility Trips (10,200 rows)
    5. HR Employee Attrition (5,000 rows)
    6. SaaS Subscription ARR (8,150 rows)
  - **Verifiable Benchmark Results:**
    - **Raw Data Quality Score:** 82.6 / 100
    - **Cleaned Data Quality Score:** **98.4 / 100** (+15.8% average gain)
    - **Validation Pass Rate:** **98.8%** across 81 evaluated checks
    - **Circuit Breaker Recovery Rate:** **83.3%** of initially failing runs saved
    - **Execution Time:** ~1.1 seconds for 50k rows
- **Where it lives in the code:** [`app/agents/reporter.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/reporter.py), [`benchmark.py`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/benchmark.py), [`BENCHMARK_RESULTS.md`](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/BENCHMARK_RESULTS.md).

---

## 2. Whiteboard Architecture Diagram

```
                             [ Raw CSV Upload ]
                                     │
                                     ▼
                           [ FastAPI Service ]
                         (X-API-Key, CORS, DB Job)
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │     LangGraph State Machine     │
                    │                                 │
                    │  [ Stage 1: Profile & Schema ]  │
                    │  (Pandas stats, inferred types) │
                    │                │                │
                    │                ▼                │
                    │      [ Stage 2: Planner ]       │
                    │  (LLM TransformationPlan JSON)  │
                    │                │                │
                    │                ▼                │
                    │     ┌─▶ [ Stage 3: Code ]       │
                    │     │   (LLM Pandas script)     │
                    │     │          │                │
                    │     │          ▼                │
                    │     │   [ Stage 4: Execute ]    │
                    │     │   (Subprocess Sandbox)    │
                    │     │          │                │
                    │     │          ▼                │
                    │     │   [ Stage 5: Validate ]   │
                    │     │   (Pandera & 7 Categories)│
                    │     │          │                │
                    │     │     Passed?               │
                    │     │     ├── No (Retry < 3) ───┘
                    │     │     │   [Circuit Breaker]
                    │     │     │   Injects Pandera Errors
                    │     │     │
                    │     │     └── Yes / Max Retries
                    │     │          │
                    │     │          ▼
                    │     │   [ Stage 6: Report ]     │
                    │     │   (Markdown Quality Report)
                    │     └───────────────────────────│
                    └─────────────────────────────────┘
                                     │
                                     ▼
                      [ Cleaned CSV & Quality Report ]
```

---

## 3. Top Interview Questions & How to Answer Them

### Q1: "Why use LangGraph instead of a standard LangChain chain or LlamaIndex pipeline?"
**Answer:**
> *"Linear pipelines work well when tasks never fail, but data engineering is inherently iterative. If an LLM writes code that causes a Pandera schema failure or introduces new nulls, a linear chain terminates with an error. LangGraph allows us to model the preprocessing workflow as a cyclic state machine. The state maintains context—the original profile, the current code, the error feedback, and a retry counter. We route conditionally from `validate` back to `code` with the exact Pandera failure cases, allowing the agent to heal its own code without user intervention."*

### Q2: "What is your Data Quality Score formula, and why is it structured that way?"
**Answer:**
> *"I designed a 4-dimensional composite score based on enterprise data observability standards:
> $$\text{DQ Score} = 0.35 \times \text{Completeness} + 0.25 \times \text{Validity} + 0.20 \times \text{Uniqueness} + 0.20 \times \text{Consistency}$$
> - **Completeness (35%):** Percent of populated, non-null cells.
> - **Validity (25%):** Conformance to Pandera column contracts, range bounds, and regex patterns.
> - **Uniqueness (20%):** Absence of exact duplicate rows and primary key collisions.
> - **Consistency (20%):** Conformance to cross-field relational rules and entity normalization.
> Across 6 benchmark datasets totaling ~50,000 rows, this metric improved from an average of 82.6/100 to 98.4/100."*

### Q3: "How does your system distinguish between an exact duplicate and a key collision?"
**Answer:**
> *"A lot of junior engineers make the mistake of running `df.drop_duplicates(subset=['id'])`. That is dangerous in production. If two rows have the same `order_id` but different prices or customer names, that's a Primary Key Collision, not a duplicate. Dropping one row silently deletes real business transactions.
> In my validation engine, I run two separate checks:
> 1. `df.duplicated()` catches full-row exact duplicates (safe to deduplicate).
> 2. `df.duplicated(subset=[pk]) & ~df.duplicated()` isolates rows that share the same key but differ in other columns.
> Key conflicts are flagged as high-severity escalations for human review rather than being arbitrarily overwritten."*

### Q4: "Why execute code in a subprocess instead of using `eval()` or `exec()`?"
**Answer:**
> *"Running LLM-generated code via `exec()` in the main application process is a severe security vulnerability and stability risk: malicious or malformed code could mutate global state, corrupt memory, or hang the web server on infinite loops. I execute the generated code in an isolated Python subprocess via `subprocess.run` with a strict 60-second timeout. Standard output and error streams are captured into structured JSON logs, and any unhandled exception exits cleanly without affecting the FastAPI application."*

---

## 4. Polished, Defensible Resume Bullets

Copy and paste these directly onto your resume:

- **Built a 5-stage LangGraph autonomous loop** (`Profile → Plan → Code → Execute → Validate`) to clean and standardize raw datasets using dynamic Pandas script execution in an isolated subprocess sandbox.
- **Engineered a self-healing circuit breaker** utilizing Pandera schema contracts to intercept failure cases (`pa.errors.SchemaErrors`) and feed structured diagnostics to LLM retries, **recovering 83.3% of failing runs**.
- **Implemented multi-category validation engine** evaluating cross-field temporal precedence, arithmetic identities, fuzzy entity deduplication, and primary key collision detection.
- **Designed mathematical Data Quality Scoring model** (0–100 across Completeness, Validity, Uniqueness, Consistency), **improving baseline quality from 82.6 to 98.4 (+15.8%)** on ~50k rows across 6 domains.
- **Developed asynchronous FastAPI backend** with job polling, API-key authentication, CORS controls, and HTTP security headers (`nosniff`, `X-Frame-Options`), outputting auto-generated Markdown quality reports.

