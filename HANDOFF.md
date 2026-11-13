# Autonomous Agentic Data Preprocessing & Quality Engineering System
## Comprehensive Project Handoff & Architecture Context

**Project Repositories:**
- **Backend:** `https://github.com/mkathir0/Agentic-Data-Preprocessing-System.git` (Local: `/Users/mithrajitkathir/Documents/agentic-ai-preprocessing`)
- **Frontend:** `https://github.com/mkathir0/Agentic-Data-Preprocessing-System-Frontend.git` (Local: `/Users/mithrajitkathir/Documents/Agentic-Data-Preprocessing-System-Frontend`)

---

## 1. Executive Summary & Core Objective

### What We Are Building
An **autonomous, enterprise-grade AI Data Preprocessing & Validation System** capable of taking raw, noisy, malformed enterprise CSVs (with millions of messy entries, inconsistent date conventions, currency characters, missing indicators, whitespace noise, sentinel values, and relational contradictions) and transforming them deterministically into production-ready analytical datasets.

### Core Philosophy
1. **Deterministic Execution over Direct LLM Data Rewriting:**  
   LLMs never touch or alter row-level tabular data directly. Instead, an LLM agent inspects high-level column distributions and sample anomalies to synthesize a strict, reproducible Python `pandas` cleaning script.
2. **Multi-Agent Orchestration with Self-Healing Circuit Breaker:**  
   Coordinated via **LangGraph**, the pipeline moves through `Profile → Plan → Code → Execute → Validate → Report`. If the execution fails or schema constraints fail, failure cases are fed back to the code generator in an automated retry loop (up to 3 attempts).
3. **Autonomous Safety Protocol (Human-in-the-Loop):**  
   If primary key collisions (identical IDs with conflicting attributes) or domain contradictions are detected, the system safely preserves the data and flags the rows for human review rather than blindly overwriting or deleting business transactions.
4. **Production Handoff Interoperability:**  
   A validated run produces these core deliverables:
   - The **Cleaned CSV** (`cleaned_{filename}.csv`)
   - The **Production Python Cleaning Script** (`clean_{filename}.py`) for zero-token ETL runs
   - The **Great Expectations Validation Suite** (`quality_suite_{filename}.json`) for automated CI/CD gating
   - The **Executive Engineering Markdown Audit** (`report_{filename}.md`) with statistical before/after diffs
   The backend also writes a machine-readable validation report (`validation_{filename}.json`). Review candidates retain validation evidence but are not marked production-approved.

### Current Backend Contract
- Automatic edits use deterministic candidates for numeric/currency parsing, unambiguous date normalization, recognized booleans, case-only categorical variants, whitespace, and exact whole-row duplicates. Imputation, fuzzy category merging, outlier correction, and contradictory business-rule reconciliation preserve source values and require review.
- Validation compares all output records and multiplicities against that deterministic policy. A file is production-approved only when integrity and validation pass with no review requirement.
- Validation failures return through planning and code generation for at most three repair attempts. Fuzzy matches are candidates for human review, never automatic merges.
- Generated scripts run with OS restrictions: macOS Seatbelt or a Linux Docker container with no network, read-only root filesystem, dropped capabilities, and resource limits. Execution fails closed when the required runtime is missing.
- Uploads create durable queue records before the API responds. A separate `python -m app.worker` process claims work with renewable leases; expired leases are reclaimable after a worker crash. Production workers must share the database and upload/output filesystem.

---

## 2. End-to-End System Architecture

```
[ Raw CSV Upload ]
        │
        ▼
[ Node 1: ProfilerAgent ] ──► Calculates distributions, null rates, regex patterns, sentinel tokens
        │
        ▼
[ Node 2: PlannerAgent ]  ──► Evaluates anomalies against domain invariants; outputs structured JSON action plan
        │
        ▼
[ Node 3: CodeGenAgent ]  ──► Generates self-contained, vectorized Python pandas script
        │
        ▼
[ Node 4: ExecutorAgent ] ──► Runs script in OS-restricted workspace; atomically publishes candidate CSV
        │
        ▼
[ Node 5: ValidationEngine ]
        ├── Dynamic Pandera DataFrameSchema (lazy=True contract check)
        ├── 7-Layer Quality Invariant Checks (Completeness, Validity, Uniqueness, Consistency)
        ├── Primary Key Collision & Fuzzy Duplicate Detection
        └── 4D Mathematical Data Quality Score (0–100)
        │
   ┌────┴─────────────────────────────┐
   │                                  │
[Checks Failed & Retries < 3]     [Checks Passed OR Retries == 3]
   │                                  │
   ▼                                  ▼
[Circuit Breaker Feedback]       [ Node 6: ReporterAgent ]
(Route back to Planner)               │
                                      ├── Generates Markdown Report (.md)
                                      ├── Exports Great Expectations Suite (.json)
                                      └── Updates Job Status (COMPLETED / REVIEW_FLAGGED)
```

---

## 3. Historical Change Log (Superseded Where Noted)

The issue notes below describe earlier behavior. The current backend contract in Section 1 takes precedence; historical claims about successful cleaning, production status, fuzzy deduplication, and benchmark results are not evidence that old output files are safe.

### Issue 1: Groq Token Limit Exceeded (HTTP 413)
* **Symptom:** `Request too large for model 'qwen/qwen3.8-27b' on input tokens per minute (ITPM): Limit 7000, Requested 10530`.
* **Root Cause:** Raw dataset profiling dumps included full value samples and lengthy JSON structures that overwhelmed the free/on-demand Groq token budget.
* **Resolution:**
  - Standardized on `llama-3.3-70b-versatile`.
  - Streamlined profiling payloads in `profiler.py` by summarizing top 3 frequent values, truncating long string representations, and compacting system prompts to consume <2,500 tokens.

### Issue 2: LLM Output JSON Validation Failures (HTTP 400)
* **Symptom:** `Failed to validate JSON. Please adjust your prompt. 'json_validate_failed'`.
* **Root Cause:** Qwen/Groq JSON modes occasionally returned unescaped regex strings or markdown code fences (` ```json `) inside raw JSON responses.
* **Resolution:**
  - Upgraded Pydantic models (`CleaningPlan`, `PlanAction`, `ProfilerOutput`) in `schemas.py`.
  - Added JSON extraction sanitizers that strip markdown fences and parse nested JSON payloads cleanly before Pydantic validation.

### Issue 3: Missing LangSmith Observability & HTTP 405 Errors
* **Symptom:** LangSmith API key was configured, but live traces were not appearing in the frontend dashboard.
* **Root Cause:** The backend endpoint `GET /api/jobs/{job_id}/langsmith-traces` was executing an raw `httpx.get("https://api.smith.langchain.com/api/v1/runs")`. The LangSmith Cloud API rejects GET requests on that endpoint with **HTTP 405 Method Not Allowed** (querying runs requires POST `/runs/query`).
* **Resolution:**
  - Migrated to the official `from langsmith import Client; client = Client(...)` SDK.
  - Decorated all pipeline nodes (`run_pipeline`, `node_profile`, `node_plan`, `node_code`, `node_execute`, `node_validate`, `node_report`) and underlying agent functions with `@traceable`.
  - Tagged runs with `agent_id = "agentic-data-preprocessing-pipeline"`, `environment = "production"`, and project `agentic-data-preprocessing-pipeline`.
  - Endpoint now streams live execution events and microsecond durations directly to the UI.

### Issue 4: Great Expectations Suite Tab Was Empty
* **Symptom:** The "Great Expectations" tab in the frontend showed 0 assertions / empty suite.
* **Root Cause:** Inside `execute_full_validation_engine` in `validation_engine.py`, `export_great_expectations_suite` failed silently with `NameError: name 'Path' is not defined` inside an exception catch block.
* **Resolution:**
  - Added `from pathlib import Path` to `validation_engine.py`.
  - Properly attached `report.great_expectations_suite` to `ValidationReport`.
  - Successfully generates and exports **26 typed assertions** including `expect_column_values_to_not_be_null`, `expect_column_values_to_be_unique`, and `expect_column_values_to_match_regex`.

### Issue 5: Pipeline Card Showed "FAILED / Pipeline Halted" Despite Successful Cleaning
* **Symptom:** The execution card showed red `AGENT PIPELINE · FAILED` with message `✗ Pipeline failed. Max retries exceeded`, yet the cleaned CSV and markdown reports existed on disk.
* **Root Cause:**
  - The dataset had 240 duplicate transaction IDs. Pandera enforced strict uniqueness on `transaction_id`, failing the contract on all 3 retries.
  - When retries reached 3, `node_report` returned `final_status = "FAILED"`, and `jobs.py` marked the job as `status = "FAILED"`.
  - Because output files were created, this was a false alarm: the system had successfully normalized the data and preserved conflicting records for safety.
* **Resolution:**
  - Updated `jobs.py`: if `output_file` exists and has size > 0, the job status is set to **`COMPLETED`** with `current_step = "REVIEW_FLAGGED"`.
  - Updated `EnterpriseAIPipeline` UI: displays **`✓ DONE` / `5/5 nodes complete`** with an amber review flag instead of a broken pipeline card.

### Issue 6: Warning Banner Displayed "0 Primary Key Collisions"
* **Symptom:** The banner displayed: *"The validation engine detected 0 primary key collisions... Human Review Recommended"*.
* **Root Cause:**
  - `detect_key_conflicts` only looked at `schema.primary_keys` (which was empty `[]`), while Pandera used a fallback heuristic that identified `transaction_id` as unique.
  - Thus, `key_conflict_count` was computed as `0`, but Pandera still failed, causing the UI to print `0 collisions` inside a warning.
* **Resolution:**
  - Unified candidate primary key detection across both Pandera and collision checks (`pks = ['transaction_id']`).
  - Now accurately detects the **240 actual primary key collisions**.
  - Updated frontend banner logic in `ResultsPanel.tsx` to accurately display the collision count when >0, state domain anomalies when collisions are 0, and show the green "100% Deterministic Cleaning" banner when zero issues occur.

### Issue 7: Missing Markdown Report Download & Viewer in UI
* **Symptom:** User was unable to find or download the `.md` report from the browser.
* **Root Cause:** The backend generated `outputs/report_{filename}.md`, but there was no download button or viewer in the frontend.
* **Resolution:**
  - Added `downloadReport(jobId, filename)` and `downloadMarkdownBlob(...)` in `src/lib/api.ts`.
  - Added an **"Export Report (.md)"** button in the dashboard header.
  - Added a dedicated **"Engineering Report (.md)"** tab in `ResultsPanel.tsx` allowing in-browser viewing, clipboard copying, and downloading.

---

## 4. Production Handoff & The 4 Next Steps

When a dataset finishes processing, users now have a **"Production Handoff & Recommended Next Steps"** card in the UI directing them across 4 clear pathways:

| # | Production Pathway | Deliverable | Action & Purpose |
|---|---|---|---|
| **1** | **Direct BI & Analytics** | `cleaned_{filename}.csv` | Load directly into Tableau, Power BI, Excel, or Python notebooks. Inconsistent dates, sentinels, and whitespaces have been eliminated. |
| **2** | **Automated ETL Scheduling** | `clean_{filename}.py` | Deploy the synthesized Python pandas script directly into Apache Airflow, Prefect, or AWS Lambda to process future raw batches at **$0 LLM token cost**. |
| **3** | **Continuous CI/CD Gate** | `quality_suite_{filename}.json` | Plug the exported Great Expectations suite into GitHub Actions or dbt tests to automatically catch vendor schema regressions. |
| **4** | **Domain Reconciliation** | Review Quarantined Rows | Navigate to the "Quality & Validation" tab to inspect the 240 primary key collisions with business stakeholders before merging into the production database. |

---

## 5. Technical Environment & Repository Details

### Backend (`agentic-ai-preprocessing`)
- **Language / Framework:** Python 3.11, FastAPI, Uvicorn, SQLite (`data_engineer.db`), SQLAlchemy.
- **Agent Stack:** LangGraph, LangSmith SDK, Groq Llama 3.3 70B, Pandera, Great Expectations, Pandas, NumPy.
- **Port:** `http://localhost:8000`
- **Key Files:**
  - [app/agents/graph.py](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/graph.py): 5-stage state graph and LangSmith `@traceable` hooks.
  - [app/agents/validation_engine.py](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/agents/validation_engine.py): Pandera contracts, GX suite generation, and 4D quality scoring.
  - [app/routers/jobs.py](file:///Users/mithrajitkathir/Documents/agentic-ai-preprocessing/app/routers/jobs.py): Job processing, report serving, and LangSmith SDK run querying.
  - `OUTPUT_DIR/<job_id>/`: Per-job cleaned or review-candidate CSV, script, engineering report, validation JSON, and optional Great Expectations suite. Historical generated artifacts were removed.

### Frontend (`Agentic-Data-Preprocessing-System-Frontend`)
- **Language / Framework:** Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS, Framer Motion, Lucide Icons.
- **Port:** `http://localhost:3000`
- **Key Files:**
  - [src/components/dashboard/ResultsPanel.tsx](file:///Users/mithrajitkathir/Documents/Agentic-Data-Preprocessing-System-Frontend/src/components/dashboard/ResultsPanel.tsx): Tabbed viewer (Data Preview, AI Recipe, Quality Audit, Great Expectations, Markdown Report, LangSmith Traces) + Next Steps card.
  - [src/components/ui/ai-agent-pipeline.tsx](file:///Users/mithrajitkathir/Documents/Agentic-Data-Preprocessing-System-Frontend/src/components/ui/ai-agent-pipeline.tsx): SVG interactive 5-stage agent flow visualization.
  - [src/lib/api.ts](file:///Users/mithrajitkathir/Documents/Agentic-Data-Preprocessing-System-Frontend/src/lib/api.ts): Typed client API for file uploads, report downloads, and trace fetching.

---

## 6. Current Verification Status

- The backend regression suite covers upload-to-worker processing, deterministic transformations, integrity validation, review-only outputs, durable queue claims/recovery, report artifacts, and OS sandbox restrictions.
- Run it with `uv run python -m unittest discover -s tests -v`.
- Historical database jobs and generated outputs from the older policy were cleared. The raw uploaded test CSVs remain unchanged.
- No live API, worker, or frontend service is assumed to be running.
