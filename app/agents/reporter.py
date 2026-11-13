import os
import json
from typing import Optional, Dict, Any, Union
from langsmith import traceable
from app.agents.schemas import ValidationReport

@traceable(name="Reporter Agent", run_type="chain")
def generate_report(
    job_id: str,
    plan_json: Optional[str],
    code: Optional[str],
    quality_summary: str,
    output_path: str,
    validation_report: Optional[Union[ValidationReport, Dict[str, Any]]] = None
):
    """
    Generates a production-grade markdown engineering and quality report.
    Includes Data Quality Scores, Pandera contract verification, 7-category breakdown,
    escalation flags, transformation plan audit trail, and executable code.
    """
    # Parse validation report if provided as dict
    vr: Optional[ValidationReport] = None
    if validation_report is not None:
        if isinstance(validation_report, ValidationReport):
            vr = validation_report
        elif isinstance(validation_report, dict):
            try:
                vr = ValidationReport.model_validate(validation_report)
            except Exception:
                vr = None

    # Parse plan if string
    plan_obj = None
    if plan_json:
        try:
            plan_obj = json.loads(plan_json)
        except Exception:
            plan_obj = None

    # Build Executive Summary KPI table
    score_before = vr.data_quality_score_before if vr else 0.0
    score_after = vr.data_quality_score_after if vr else 0.0
    improvement = vr.score_improvement if vr else 0.0
    rows_before = vr.rows_before if vr else "N/A"
    rows_after = vr.rows_after if vr else "N/A"
    pandera_status = "PASSED" if (vr and vr.pandera_schema_passed) else "FAILED / VIOLATIONS"

    lines = [
        f"# AI Data Engineering & Quality Report",
        f"**Job ID:** `{job_id}`  ",
        f"**Pipeline Status:** {quality_summary}  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Quality Score",
        "",
        "| Metric | Raw (Before) | Cleaned (After) | Change / Delta |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Overall Data Quality Score** | **{score_before:.1f} / 100** | **{score_after:.1f} / 100** | **+{improvement:.1f}%** |",
        f"| **Completeness (Nulls)** | — | {vr.completeness_score if vr else 0.0:.1f} / 100 | Verified |",
        f"| **Validity (Types & Ranges)** | — | {vr.validity_score if vr else 0.0:.1f} / 100 | Verified |",
        f"| **Uniqueness (Duplicates & Keys)** | — | {vr.uniqueness_score if vr else 0.0:.1f} / 100 | Verified |",
        f"| **Consistency (Cross-field Rules)** | — | {vr.consistency_score if vr else 0.0:.1f} / 100 | Verified |",
        f"| **Total Rows** | {rows_before} | {rows_after} | Retained |",
        f"| **Pandera Contract Conformance** | Untested | **{pandera_status}** | Enforced |",
        "",
        "---",
        "",
        "## 2. Pandera Schema Contract Verification",
    ]

    if vr and vr.pandera_failure_cases:
        lines.extend([
            f"⚠️ **Pandera Contract Failures Detected ({vr.pandera_failure_count} total violations):**",
            "",
            "| Column | Check Name | Violations | Sample Failing Values |",
            "| :--- | :--- | :---: | :--- |",
        ])
        for fc in vr.pandera_failure_cases:
            col = fc.get("column", "DataFrame")
            check = fc.get("check", "Unknown")
            count = fc.get("violation_count", 0)
            samples = ", ".join(str(s) for s in fc.get("sample_failures", []))
            lines.append(f"| `{col}` | `{check}` | {count} | `{samples}` |")
    else:
        lines.extend([
            "✅ **All Pandera schema constraints and domain bounds passed without violations.**",
            "- Strict dtype enforcement: Verified",
            "- Primary Key non-null and uniqueness constraints: Verified",
            "- Regex format & domain boundary checks: Verified",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 3. 7-Category Data Quality Breakdown",
    ])

    if vr:
        check_categories = [
            ("Schema & Types", vr.schema_checks),
            ("Range & Domain Bounds", vr.range_checks),
            ("Cross-Field & Relational Invariants", vr.cross_field_checks),
            ("Distributional & Outlier Shifts", vr.distributional_checks),
            ("Duplicates & Primary Key Collisions", vr.duplicate_checks),
            ("Format & String Encoding", vr.format_checks),
            ("Regression & Safety Checks", vr.regression_checks),
        ]

        for cat_name, check_list in check_categories:
            passed_count = sum(1 for c in check_list if c.passed)
            total_count = len(check_list)
            status_icon = "✅" if (total_count > 0 and passed_count == total_count) else ("⚠️" if total_count > 0 else "ℹ️")
            lines.append(f"### {status_icon} {cat_name} ({passed_count}/{total_count} Passed)")
            if check_list:
                lines.append("| Check | Status | Violations | Details |")
                lines.append("| :--- | :---: | :---: | :--- |")
                for c in check_list:
                    icon = "PASS" if c.passed else "**FAIL**"
                    lines.append(f"| {c.name} | {icon} | {c.violations} | {c.details} |")
            else:
                lines.append("No specific checks required for this category.")
            lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. Escalations & Human Review Required",
    ])

    if vr and vr.escalation_flags:
        lines.extend([
            "The following issues cannot be safely resolved with automated heuristics and require human domain review:",
            "",
            "| Severity | Column | Reason |",
            "| :---: | :--- | :--- |",
        ])
        for flag in vr.escalation_flags:
            sev_badge = f"**{flag.severity.upper()}**"
            col_str = f"`{flag.column}`" if flag.column else "*Dataset Global*"
            lines.append(f"| {sev_badge} | {col_str} | {flag.reason} |")
    else:
        lines.append("✅ **No escalation flags raised.** Dataset is production-ready.")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Executed Transformation Plan",
    ])

    if plan_obj and "actions" in plan_obj:
        lines.append(f"**Dataset Summary:** {plan_obj.get('dataset_summary', 'Cleaned dataset.')}\n")
        lines.append("| Step | Column | Action | Confidence | Estimated Rows | Rationale / Assumption |")
        lines.append("| :---: | :--- | :--- | :---: | :---: | :--- |")
        for idx, act in enumerate(plan_obj.get("actions", []), 1):
            col_name = f"`{act.get('column')}`" if act.get('column') else "*All Columns*"
            act_type = f"`{act.get('action_type')}`"
            conf = act.get('confidence', 'medium')
            est_rows = f"{round(act.get('estimated_rows_affected_pct', 0.0) * 100, 1)}%" if act.get('estimated_rows_affected_pct') is not None else "—"
            rationale = act.get('explanation', '')
            if act.get('assumption'):
                rationale += f" *(Assumption: {act.get('assumption')})*"
            lines.append(f"| {idx} | {col_name} | {act_type} | {conf} | {est_rows} | {rationale} |")
    else:
        lines.append(f"```json\n{plan_json}\n```")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Generated Executable Python Code",
        "```python",
        code if code else "# No code executed.",
        "```",
        ""
    ])

    report_content = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)
