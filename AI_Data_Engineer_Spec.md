# AI Data Engineer
### An Autonomous Multi-Agent Data Engineering System

---

## 1. Vision

Build an autonomous AI Data Engineer capable of taking a completely raw dataset and transforming it into a production-ready data package.

Instead of simply answering questions or generating code from a prompt, the system behaves like a **junior data engineer**: it continuously observes, plans, executes, validates, and improves its work until the dataset satisfies engineering quality standards.

> **Not AutoML.** The focus is Data Engineering, not model training.

---

## 2. Problem Statement

When a company receives a new dataset, a data engineer typically spends hours on repetitive tasks:

- Understanding the dataset
- Profiling data
- Detecting data quality issues
- Cleaning data
- Standardizing formats
- Designing schemas
- Creating transformations
- Writing SQL
- Building dbt models
- Writing validation tests
- Creating documentation

Most of these steps follow predictable engineering patterns — which makes them automatable.

---

## 3. Core Philosophy

**This project is NOT:** *"Ask ChatGPT to clean my CSV."*

**This project IS:** The AI actually performing engineering work, where every decision either:
- comes from **deterministic analysis**, or
- is **reasoning grounded in deterministic evidence**

> **Key Design Principle**
> Deterministic systems collect evidence. LLMs make engineering decisions.
> *(Not: LLMs trying to do everything.)*

---

## 4. Architecture

```
                          User uploads CSV
                                 │
                                 ▼
                          FastAPI Backend
                                 │
                                 ▼
                       LangGraph Orchestrator
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
       Dataset Profiler     Quality Agent       Schema Agent
              │                  │                  │
        ydata-profiling   Great Expectations      Pandas
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                    Transformation Planner (LLM)
                                 │
                ┌───────────────┼────────────────┐
                ▼                ▼                ▼
        Python Generator   SQL Generator     dbt Generator
                │
                ▼
        Execute Pipeline (Docker/Python Sandbox)
                │
                ▼
          Validation Agent
                │
            Pass? ──────────────── No
                │                   │
               Yes                  ▼
                │            Back to Planner
                ▼
          Final Artifacts
```

**Why LangGraph?**
LangGraph is not responsible for cleaning data — its job is **orchestration**. It manages the stateful loop: Observe → Plan → Execute → Validate → Retry → Finish. Unlike LangChain's simple linear chains, LangGraph allows loops and stateful workflows, which this project fundamentally needs.

---

## 5. Multi-Agent Breakdown

### 5.1 Dataset Profiler Agent
**Responsibilities:** inspect dataset, infer datatypes, generate statistics, detect cardinality, calculate distributions, identify categorical/numerical columns.

**Tools:** Pandas, Polars, ydata-profiling

**Output (JSON):**
```json
{
  "rows": 500000,
  "columns": 42,
  "missing": { "Age": 14 }
}
```

### 5.2 Data Quality Agent
**Responsibilities:** missing values, duplicate rows, invalid values, impossible values, inconsistent formatting, mixed datatypes.

**Tools:** Great Expectations, Pandera

**Output:** Structured quality report

### 5.3 Schema Agent
**Responsibilities:** infer primary keys, candidate foreign keys, datetime columns, categorical columns, identifier columns.

**Future scope:** infer fact tables, dimension tables, relationships.

### 5.4 Transformation Planner (LLM)
**Input:** ONLY structured outputs from the previous three agents — **never raw CSV**.

**Responsibilities:** decide missing value strategy, duplicate strategy, datatype conversions, normalization, feature standardization, pipeline ordering.

**Example reasoning:**
| Column | Decision | Reason |
|---|---|---|
| `Age` | Median imputation | Distribution is right-skewed |
| `Country` | Normalize spellings | 97 unique spellings detected |

### 5.5 Code Generation Agent
Generates Python, SQL, dbt models, and transformation scripts. **No execution happens here** — generation only.

### 5.6 Execution Agent
Runs the generated pipeline in a **Docker sandbox** or **Python subprocess**.

**Output:** new cleaned dataset

### 5.7 Validation Agent
Runs Great Expectations, Pandera, and custom validation checks: unique IDs, no missing keys, valid ranges, schema conformity, etc.

---

## 6. The Agent Loop

```
Observe → Analyze → Plan → Generate Code → Execute → Validate
                                                          │
                                                       Pass?
                                                          │
                                          ┌───No──────────┴──────────Yes───┐
                                          ▼                                ▼
                            Planner receives validation report        Finish
                                          │
                                Generate improved plan
                                          │
                                    Execute Again
                                          │
                                       (Repeat)
```

This loop — not the prompting — is what makes the project **agentic** rather than prompt engineering.

---

## 7. Human Escalation

The system must **not** blindly retry forever. If confidence stays low or multiple retries fail, it flags **Human Review Required**.

**Example:**
> `ZIP Code` — 40% missing. Cannot determine safe strategy.
> **Recommendation:** Review manually.

---

## 8. Why Existing Tools Aren't Enough

| Tool | Solves |
|---|---|
| ydata-profiling | Only profiling |
| Great Expectations | Only validation |
| Copilot | Only generates code |
| AutoML | Only optimizes ML pipelines |

**None of them orchestrate the complete engineering workflow.** That orchestration + retry loop is the differentiator.

---

## 9. Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI |
| Frontend | React |
| State | PostgreSQL |
| Agent Framework | LangGraph |
| LLM Router | LiteLLM |
| LLM | Claude / GPT-4.1 / Gemini |
| Processing | Pandas, Polars |
| Validation | Great Expectations, Pandera |
| Profiling | ydata-profiling |
| Execution | Docker Sandbox |
| SQL Parsing | sqlglot |
| Documentation | Markdown + Jinja |
| Vector DB (future) | Qdrant |

---

## 10. Roadmap

### MVP — Single CSV
- ✔ Profile dataset
- ✔ Detect quality issues
- ✔ Build cleaning strategy
- ✔ Generate Python cleaning code
- ✔ Execute cleaning
- ✔ Validate cleaned data
- ✔ Produce documentation

**Outputs:** `cleaned.csv`, `clean_pipeline.py`, `engineering_report.md`, `validation_report.md`

### V2 — Multi-Dataset
Multiple datasets, relationship inference, primary key detection, foreign key inference, automatic joins.

### V3 — SQL & dbt
SQL transformations, dbt project, YAML/schema files, Great Expectations suite.

### V4 — Lineage & Contracts
Dataset lineage, impact analysis, data contracts, pipeline documentation.

### V5 — Orchestration & Deployment
Airflow DAG generation, Dagster assets, Prefect flows, cloud deployment.

---

## 11. Project Differentiator

**Traditional AI:** Prompt → Generate Code → Done

**This Project:** Observe → Reason → Generate → Execute → Validate → Improve → Repeat

The AI behaves like an engineer, not a chatbot.

---

## 12. Long-Term Vision

Evolve from an autonomous dataset cleaner into a full AI Data Engineer capable of:

- Designing ETL pipelines
- Building dbt projects
- Generating SQL transformations
- Creating validation suites
- Producing engineering documentation
- Understanding dataset relationships
- Performing lineage analysis
- Automatically iterating until quality requirements are met

**End goal:** it should feel like hiring a junior data engineer that autonomously handles repetitive engineering work — and knows when to escalate ambiguous decisions to a human.

---

## 13. Build Order (for agent handoff)

A sensible sequence to hand this to a coding agent, phase by phase:

1. **Scaffolding** — FastAPI backend, React frontend shell, PostgreSQL schema for job/run state
2. **Profiler Agent** — CSV upload → Pandas/ydata-profiling → JSON profile output
3. **Quality Agent** — Great Expectations/Pandera checks → structured quality report
4. **Schema Agent** — key/type inference on top of profile output
5. **LangGraph Orchestrator** — wire Observe → Plan → Execute → Validate loop with retry state
6. **Transformation Planner (LLM)** — consumes only structured JSON from steps 2–4, outputs a transformation plan
7. **Code Generation Agent** — plan → Python cleaning script (SQL/dbt deferred to V3)
8. **Execution Agent** — Docker sandbox/subprocess runner, capture output dataset
9. **Validation Agent** — re-run quality checks against output, pass/fail signal back to orchestrator
10. **Escalation logic** — retry cap + confidence threshold → "Human Review Required" flag
11. **Reporting** — Jinja/Markdown generation of `engineering_report.md` and `validation_report.md`

