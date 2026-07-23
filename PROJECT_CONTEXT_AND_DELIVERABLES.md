# AI Data Engineer: Context & Deliverables

## 1. Project Vision & Context
The goal of this project is to build an autonomous AI Data Engineer capable of taking a completely raw dataset (CSV) and transforming it into a production-ready data package. 
Unlike simple LLM wrappers, this system acts like a junior data engineer: it **observes, plans, executes, validates, and improves** its work until engineering quality standards are met. It relies on deterministic tools (like Pandas and Great Expectations) for evidence and LLMs purely for decision-making.

## 2. Architecture & Tech Stack
- **Backend:** FastAPI, Python (using `uv` for environment management)
- **Frontend:** React
- **Orchestration:** LangGraph (stateful loops for Observe -> Plan -> Execute -> Validate)
- **LLM Routing:** LiteLLM (Claude / GPT-4o / Gemini)
- **Data Tools:** Pandas, Polars, ydata-profiling, Great Expectations, Pandera, sqlglot
- **State/Database:** PostgreSQL / SQLite
- **Execution Environment:** Docker Sandbox or Python Subprocess

## 3. The 5 Deliverables

### Deliverable 1: Foundation & Infrastructure
*Goal: Get the core backend, database, and frontend shell running to handle file uploads and job states.*
- **Backend Task:** Create FastAPI routes for uploading a CSV and checking job status. Set up a PostgreSQL/SQLite database to track job runs.
- **Frontend Task:** Create a React application with a clean UI to upload a CSV file, trigger a cleaning job, and poll/display job status.

### Deliverable 2: The Data Analysis Agents
*Goal: Build the agents that observe and analyze the raw data deterministically, without relying on LLM hallucinations.*
- **Profiler Agent:** Takes a CSV, runs `ydata-profiling` or Pandas, and outputs a structured JSON profile (rows, missing values, distributions).
- **Quality Agent:** Runs `Great Expectations` or `Pandera` against the CSV to generate a structured report on anomalies.
- **Schema Agent:** Infers primary keys, candidate foreign keys, and column types.

### Deliverable 3: Planning & Code Generation
*Goal: Use an LLM to read the structured outputs from Deliverable 2 and generate cleaning code.*
- **Transformation Planner (LLM):** Takes the JSON outputs from the analysis agents. Outputs a structured JSON/Pydantic "Plan" detailing what needs to be cleaned and how.
- **Code Generation Agent:** Takes the plan and generates a standalone, executable Python script designed to clean the dataset (no execution happens here).

### Deliverable 4: Execution & Validation
*Goal: Safely execute the generated code and verify the results.*
- **Execution Agent:** Runs the generated Python script against the raw CSV in an isolated Python subprocess or Docker sandbox. Outputs `cleaned.csv`.
- **Validation Agent:** Re-runs the `Great Expectations` checks against the cleaned CSV to verify improvements and ensure the script did not break the data.

### Deliverable 5: Orchestration & Reporting
*Goal: Tie all agents into a continuous, autonomous loop.*
- **LangGraph Orchestrator:** Wire the agents together into the core loop: `Observe → Plan → Generate → Execute → Validate`.
- **Escalation Logic:** Add a circuit breaker to loop back to the Planner if validation fails, and flag for "Human Review Required" if retries are exhausted.
- **Reporting Agent:** Generate final `engineering_report.md` and `validation_report.md` artifacts.

---
**Note for Agents:** If you are reading this file, you have been spun up to work on a specific part of the architecture in parallel. Check your prompt to see which Deliverable and which component (Frontend or Backend) you are assigned to. Ensure your work cleanly integrates with the architecture described above.
