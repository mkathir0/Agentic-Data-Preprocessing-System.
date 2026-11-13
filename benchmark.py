"""Measure the conservative cleaning policy without mutating benchmark sources.

This is a deterministic policy benchmark, not an evaluation of LLM quality or
production readiness. Every candidate is checked against the same supported
actions and record-level integrity contract as uploaded jobs.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from app.agents.cleaning_policy import apply_cleaning_plan
from app.agents.validation_engine import execute_full_validation_engine
from app.utils.data_loader import load_dataframe


BENCHMARK_FILES = [
    "bikeshare_trips.csv",
    "ecommerce_orders.csv",
    "fintech_transactions.csv",
    "healthcare_patients.csv",
    "hr_employee_attrition.csv",
    "saas_churn_metrics.csv",
]
SUPPORTED_PLAN = [
    {"action_type": "strip_whitespace", "column": None, "confidence": "high"},
    {"action_type": "drop_duplicates", "column": None, "confidence": "high"},
]


def run_benchmarks(
    benchmark_dir: str | Path = "benchmark_data",
    summary_path: str | Path = "benchmark_summary.json",
    report_path: str | Path = "BENCHMARK_RESULTS.md",
    benchmark_files: list[str] | None = None,
) -> dict:
    results = []
    root = Path(benchmark_dir)

    for filename in benchmark_files or BENCHMARK_FILES:
        input_path = root / filename
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        before = load_dataframe(str(input_path))
        candidate, review_reasons = apply_cleaning_plan(before, SUPPORTED_PLAN)
        with tempfile.TemporaryDirectory(prefix="policy-benchmark-") as directory:
            output_path = Path(directory) / "candidate.csv"
            candidate.to_csv(output_path, index=False)
            report = execute_full_validation_engine(
                input_path=str(input_path),
                output_path=str(output_path),
                plan_actions=SUPPORTED_PLAN,
            )

        if not report.integrity_passed:
            raise RuntimeError(f"Integrity verification failed for {filename}; candidate rejected.")

        results.append({
            "dataset": filename,
            "rows_before": report.rows_before,
            "rows_after": report.rows_after,
            "data_quality_score_before": report.data_quality_score_before,
            "data_quality_score_after": report.data_quality_score_after,
            "score_delta": report.score_improvement,
            "checks_passed": report.total_passed,
            "checks_total": report.total_checks,
            "integrity_passed": report.integrity_passed,
            "validation_passed": report.overall_passed,
            "review_required": report.review_required or bool(review_reasons),
            "review_reasons": list(dict.fromkeys(report.review_reasons + review_reasons)),
        })

    summary = {
        "benchmark_type": "deterministic_allowlisted_policy",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_approval": False,
        "note": "Benchmark candidates are temporary. This report does not approve benchmark data for production.",
        "datasets": results,
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, Path(report_path))
    return summary


def _write_markdown(summary: dict, report_path: Path) -> None:
    lines = [
        "# Deterministic Cleaning Policy Benchmark",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "> This measures only the allowlisted whitespace and exact whole-row deduplication policy. It does not measure LLM recovery or approve data for production.",
        "",
        "| Dataset | Rows before | Rows after | Integrity | Validation | Review required | Score delta |",
        "|---|---:|---:|:---:|:---:|:---:|---:|",
    ]
    for item in summary["datasets"]:
        lines.append(
            f"| {item['dataset']} | {item['rows_before']} | {item['rows_after']} | "
            f"{item['integrity_passed']} | {item['validation_passed']} | "
            f"{item['review_required']} | {item['score_delta']:+.2f} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run_benchmarks()
    for dataset in result["datasets"]:
        print(
            f"{dataset['dataset']}: integrity={dataset['integrity_passed']}, "
            f"validation={dataset['validation_passed']}, review={dataset['review_required']}"
        )
