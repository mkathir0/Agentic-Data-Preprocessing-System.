# Current Backend Implementation Guide

This guide describes the code as it exists now. Earlier benchmark numbers and generated samples were withdrawn because they came from transformations that bypassed the current safety policy.

## Workflow

1. `POST /api/jobs/upload` writes each source file under a job-specific directory and commits a durable `JobTask` record in the same database transaction.
2. `python -m app.worker` claims queued work using an atomic database lease. It renews the lease during processing; an expired lease can be reclaimed after a worker crash.
3. LangGraph profiles the file, plans supported actions, generates code, executes it in the OS sandbox, validates the candidate, and writes reports.
4. A job is `COMPLETED` only when the current run has both `integrity_passed=true` and `overall_passed=true`, with no human review required.
5. Ambiguous but record-preserving results remain review candidates. Their downloads are labeled `review_candidate_`; they are not production-approved outputs.

## Data safety contract

- Automatic edits include deterministic numeric/currency parsing, unambiguous date normalization, recognized boolean normalization, case-only category unification, whitespace trimming, and exact whole-row deduplication. Ambiguous or contradictory values remain for human review.
- Imputation, semantic correction, date conversion, clipping, column removal, and key-based deduplication require a reviewed domain rule. Until such a rule exists, source values and rows are preserved and the job is flagged for review.
- Validation checks record values and row multiplicities against the deterministic plan. Unplanned changes fail integrity verification.
- The planner sees structured profile, quality, and schema evidence; the execution code receives the dataset locally inside the sandbox.

## Execution isolation

- macOS runs generated code under Seatbelt with access limited to a temporary workspace and the Python runtime. Network access and execution of other programs are denied by default.
- Linux uses `Dockerfile.executor` with networking disabled, read-only root filesystem, dropped capabilities, no-new-privileges, and CPU, memory, and process limits.
- The executor fails closed if its platform sandbox is unavailable. The Linux worker needs access to the Docker daemon and the configured image.

## Validation and outputs

Current per-job artifacts are stored under `OUTPUT_DIR/<job_id>/`: cleaned or review candidate CSV, generated script, Markdown engineering report, JSON validation report, and Great Expectations suite when a contract is available. The API serves reports and CSVs from paths recorded on that job.

Trust checks live in `tests/test_data_integrity.py` and `tests/test_upload_contract.py`. They cover parser preservation, plan policy, data mutation detection, sandbox behavior, upload-to-worker processing, lease recovery, validation status, and download gating.

```bash
uv run python -m unittest discover -s tests -v
```

## Benchmark status

The previous numeric benchmark claims and generated cleaned datasets were removed. `benchmark.py` remains a narrow deterministic benchmark for whitespace trimming and exact deduplication; it validates integrity, keeps candidates temporary, and does not claim LLM recovery or production approval. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).
