import os
import json
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langsmith import traceable

from app.agents.profiler import run_profiler
from app.agents.quality import run_quality_check, run_full_validation
from app.agents.schema_agent import run_schema_inference
from app.agents.planner import generate_transformation_plan
from app.agents.code_generator import generate_cleaning_code
from app.agents.executor import execute_cleaning_code
from app.agents.reporter import generate_report
from app.agents.schemas import TransformationPlan, SchemaOutput

class AgentState(TypedDict):
    job_id: str
    input_file: str
    output_file: str
    report_file: str
    
    # Context (Phase 1 & 2)
    profiler_data: Optional[dict]
    quality_data: Optional[dict]
    schema_data: Optional[dict]
    plan_json: Optional[str]
    
    # State tracking & Self-Healing Loop (Phase 3 & 4)
    current_code: Optional[str]
    error_feedback: Optional[str]
    retry_count: int
    quality_summary: Optional[str]
    
    # Validation & Final Status (Phase 5 & Report)
    final_status: str
    quality_after: Optional[dict]
    validation_report: Optional[dict]


# ── Stage 1: Profile ─────────────────────────────────────────────────────────
def node_profile(state: AgentState) -> dict:
    """Profiles the raw dataset: computes statistics, infers schema, and scans initial anomalies."""
    input_file = state["input_file"]
    
    profiler_data = run_profiler(input_file)
    quality_data = run_quality_check(input_file)
    schema_data = run_schema_inference(input_file)
    
    return {
        "profiler_data": profiler_data.model_dump() if hasattr(profiler_data, "model_dump") else profiler_data,
        "quality_data": quality_data.model_dump(),
        "schema_data": schema_data.model_dump() if hasattr(schema_data, "model_dump") else schema_data,
    }


# ── Stage 2: Plan ────────────────────────────────────────────────────────────
def node_plan(state: AgentState) -> dict:
    """Uses LLM to create a structured TransformationPlan based purely on deterministic profiling evidence."""
    from app.agents.schemas import ProfilerOutput, QualityOutput, SchemaOutput
    
    p_data = ProfilerOutput.model_validate(state["profiler_data"])
    q_data = QualityOutput.model_validate(state["quality_data"])
    s_data = SchemaOutput.model_validate(state["schema_data"])
    
    plan = generate_transformation_plan(p_data, q_data, s_data)
    
    return {
        "plan_json": plan.model_dump_json()
    }


# ── Stage 3: Code (Generate) ────────────────────────────────────────────────
def node_code(state: AgentState) -> dict:
    """Generates executable Python pandas code to satisfy the plan. Receives Pandera feedback if retrying."""
    plan_obj = TransformationPlan.model_validate_json(state["plan_json"])
    
    code_output = generate_cleaning_code(
        plan=plan_obj,
        error_feedback=state.get("error_feedback"),
        previous_code=state.get("current_code")
    )
    
    return {
        "current_code": code_output.code
    }


# ── Stage 4: Execute ────────────────────────────────────────────────────────
def node_execute(state: AgentState) -> dict:
    """Safely executes the generated code in an isolated subprocess against the input dataset."""
    code = state["current_code"]
    exec_result = execute_cleaning_code(code, state["input_file"], state["output_file"])
    
    if not exec_result["success"]:
        return {
            "error_feedback": f"Execution failed: {exec_result['error']}",
            "retry_count": state["retry_count"] + 1
        }
    else:
        return {
            "error_feedback": None
        }


# ── Stage 5: Validate ───────────────────────────────────────────────────────
def node_validate(state: AgentState) -> dict:
    """
    Validates cleaned output using Pandera schema contracts, 7 quality invariant categories,
    fuzzy deduplication, key-conflict detection, and before/after quality scoring.
    """
    if state.get("error_feedback") and not os.path.exists(state["output_file"]):
        return state
        
    schema_obj = SchemaOutput.model_validate(state["schema_data"]) if state.get("schema_data") else None
    plan_dict = json.loads(state["plan_json"]) if state.get("plan_json") else {}
    plan_actions = plan_dict.get("actions", [])
    
    report = run_full_validation(
        input_path=state["input_file"],
        output_path=state["output_file"],
        schema=schema_obj,
        profiler_before=state.get("profiler_data"),
        plan_actions=plan_actions
    )
    
    report_dict = report.model_dump()
    high_sev = [flag.model_dump() for flag in report.escalation_flags if flag.severity == "high"]
    has_pandera_errors = not report.pandera_schema_passed
    
    # If high severity violations or Pandera schema errors exist, trigger the circuit breaker
    if not report.overall_passed and (high_sev or has_pandera_errors):
        feedback_parts = []
        if has_pandera_errors and report.pandera_failure_cases:
            feedback_parts.append("PANDERA SCHEMA FAILURES:")
            for fc in report.pandera_failure_cases:
                feedback_parts.append(
                    f"  - Column '{fc['column']}': check '{fc['check']}' failed on {fc['violation_count']} rows "
                    f"(sample failing values: {fc['sample_failures']})"
                )
        if report.new_nulls_introduced:
            feedback_parts.append(f"REGRESSION: Cleaning code introduced new nulls: {report.new_nulls_introduced}")
        if report.key_conflict_count > 0:
            feedback_parts.append(f"PRIMARY KEY COLLISION: {report.key_conflict_count} rows violate PK uniqueness: {report.key_conflict_details}")
        if high_sev:
            feedback_parts.append(f"HIGH SEVERITY ANOMALIES: {[f['reason'] for f in high_sev]}")
            
        formatted_feedback = "\n".join(feedback_parts)
        if len(formatted_feedback) > 2000:
            formatted_feedback = formatted_feedback[:2000] + "\n... [truncated to preserve token budget]"
        
        return {
            "error_feedback": formatted_feedback,
            "retry_count": state["retry_count"] + 1,
            "quality_summary": f"Validation failed with {len(report.escalation_flags)} issues (Pandera passed: {report.pandera_schema_passed}).",
            "validation_report": report_dict,
            "quality_after": {"is_valid": False, "summary": "Validation failed", "anomalies": high_sev}
        }
        
    return {
        "error_feedback": None,
        "quality_summary": f"Passed {report.total_passed}/{report.total_checks} checks (Quality Score: {report.data_quality_score_after:.1f}/100, +{report.score_improvement:.1f}%).",
        "validation_report": report_dict,
        "quality_after": {"is_valid": report.overall_passed, "summary": "Full validation complete", "anomalies": []}
    }


# ── Final Stage: Report ──────────────────────────────────────────────────────
def node_report(state: AgentState) -> dict:
    """Generates the final comprehensive Markdown engineering and quality report."""
    if state.get("error_feedback") and state["retry_count"] >= 3:
        status = "FAILED"
    else:
        status = "COMPLETED"
        
    generate_report(
        job_id=state["job_id"],
        plan_json=state.get("plan_json"),
        code=state.get("current_code"),
        quality_summary=state.get("quality_summary", "Execution completed."),
        output_path=state["report_file"],
        validation_report=state.get("validation_report")
    )
    
    return {"final_status": status}


# ── Routing Functions (Circuit Breaker) ──────────────────────────────────────
def route_after_execution(state: AgentState):
    if state.get("error_feedback"):
        if state["retry_count"] >= 3:
            return "report"
        return "code"
    return "validate"

def route_after_validation(state: AgentState):
    if state.get("error_feedback"):
        if state["retry_count"] >= 3:
            return "report"
        return "code"
    return "report"


# ── Build 5-Stage LangGraph Loop ─────────────────────────────────────────────
def build_graph():
    """
    Constructs the 5-Stage LangGraph State Machine:
    Profile → Plan → Code → Execute → Validate (with self-healing circuit breaker) → Report
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("profile", node_profile)
    workflow.add_node("plan", node_plan)
    workflow.add_node("code", node_code)
    workflow.add_node("execute", node_execute)
    workflow.add_node("validate", node_validate)
    workflow.add_node("report", node_report)
    
    workflow.set_entry_point("profile")
    
    workflow.add_edge("profile", "plan")
    workflow.add_edge("plan", "code")
    workflow.add_edge("code", "execute")
    
    workflow.add_conditional_edges(
        "execute",
        route_after_execution,
        {
            "validate": "validate",
            "code": "code",
            "report": "report"
        }
    )
    
    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "code": "code",
            "report": "report"
        }
    )
    
    workflow.add_edge("report", END)
    
    return workflow.compile()


@traceable(run_type="chain", name="Data Preprocessing Pipeline")
def run_pipeline(job_id: str, input_file: str, output_file: str, report_file: str) -> dict:
    """Top-level traceable entry point. Orchestrates the 5-stage loop."""
    graph = build_graph()
    initial_state: AgentState = {
        "job_id": job_id,
        "input_file": input_file,
        "output_file": output_file,
        "report_file": report_file,
        "profiler_data": None,
        "quality_data": None,
        "schema_data": None,
        "plan_json": None,
        "current_code": None,
        "error_feedback": None,
        "retry_count": 0,
        "quality_summary": None,
        "quality_after": None,
        "validation_report": None,
        "final_status": "PROCESSING",
    }
    result = graph.invoke(initial_state)
    return {
        "final_status": result.get("final_status", "COMPLETED"),
        "profiler_data": result.get("profiler_data"),
        "quality_data": result.get("quality_data"),
        "quality_after": result.get("quality_after"),
        "validation_report": result.get("validation_report"),
        "plan_json": result.get("plan_json"),
        "retry_count": result.get("retry_count", 0),
        "quality_summary": result.get("quality_summary"),
    }
