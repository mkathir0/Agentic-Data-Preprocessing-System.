import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from app.agents.profiler import run_profiler
from app.agents.quality import run_quality_check, run_full_validation
from app.agents.schema_agent import run_schema_inference
from app.agents.planner import generate_transformation_plan
from app.agents.code_generator import generate_cleaning_code
from app.agents.executor import execute_cleaning_code
from app.agents.reporter import generate_report

class AgentState(TypedDict):
    job_id: str
    input_file: str
    output_file: str
    report_file: str
    
    # Context
    profiler_data: Optional[dict]
    quality_data: Optional[dict]
    schema_data: Optional[dict]
    
    # State tracking
    plan_json: Optional[str]
    current_code: Optional[str]
    error_feedback: Optional[str]
    retry_count: int
    quality_summary: Optional[str]
    
    # Status tracking
    final_status: str
    quality_after: Optional[dict]
    validation_report: Optional[dict]

# 1. Profile and Plan
def node_profile_and_plan(state: AgentState) -> dict:
    input_file = state["input_file"]
    
    profiler_data = run_profiler(input_file)
    quality_data = run_quality_check(input_file)
    schema_data = run_schema_inference(input_file)
    
    plan = generate_transformation_plan(profiler_data, quality_data, schema_data)
    
    return {
        "profiler_data": profiler_data,
        "quality_data": quality_data.model_dump(),
        "schema_data": schema_data,
        "plan_json": plan.model_dump_json(),
        "plan_obj": plan # Note: Pydantic object
    }

# 2. Generate and Execute
def node_generate_and_execute(state: AgentState) -> dict:
    import json
    from app.agents.schemas import TransformationPlan
    
    plan_obj = TransformationPlan.model_validate_json(state["plan_json"])
    
    code_output = generate_cleaning_code(
        plan=plan_obj, 
        error_feedback=state.get("error_feedback"), 
        previous_code=state.get("current_code")
    )
    
    code = code_output.code
    exec_result = execute_cleaning_code(code, state["input_file"], state["output_file"])
    
    if not exec_result["success"]:
        return {
            "current_code": code,
            "error_feedback": f"Execution failed: {exec_result['error']}",
            "retry_count": state["retry_count"] + 1
        }
    else:
        return {
            "current_code": code,
            "error_feedback": None # Successfully executed
        }

# 3. Validate
def node_validate(state: AgentState) -> dict:
    import json
    from app.agents.schemas import SchemaOutput
    
    # If code crashed during execution, don't try to validate the non-existent output
    if state.get("error_feedback"):
        return state
        
    schema_obj = SchemaOutput.model_validate(state["schema_data"]) if state.get("schema_data") else None
    plan_dict = json.loads(state["plan_json"]) if state.get("plan_json") else {}
    plan_actions = plan_dict.get("actions", [])
    
    # Run full 7-category validation
    report = run_full_validation(
        input_path=state["input_file"],
        output_path=state["output_file"],
        schema=schema_obj,
        profiler_before=state.get("profiler_data"),
        plan_actions=plan_actions
    )
    
    if not report.overall_passed:
        high_sev = [flag.model_dump() for flag in report.escalation_flags if flag.severity == "high"]
        # If there are high severity flags, we consider it a failure requiring rewrite
        if high_sev:
            return {
                "error_feedback": f"Validation failed with high severity issues: {str(high_sev)}. Fix the code.",
                "retry_count": state["retry_count"] + 1,
                "quality_summary": "High severity anomalies remain.",
                "validation_report": report.model_dump()
            }
            
    return {
        "error_feedback": None,
        "quality_summary": f"Passed {report.total_passed}/{report.total_checks} checks.",
        "validation_report": report.model_dump(),
        # For backward compatibility with older logic just in case:
        "quality_after": {"is_valid": report.overall_passed, "summary": "Full validation complete", "anomalies": []}
    }

# 4. Report
def node_report(state: AgentState) -> dict:
    if state.get("error_feedback") and state["retry_count"] >= 3:
        status = "FAILED"
    else:
        status = "COMPLETED"
        
    generate_report(
        job_id=state["job_id"],
        plan_json=state["plan_json"],
        code=state["current_code"],
        quality_summary=state.get("quality_summary", "Execution failed before validation."),
        output_path=state["report_file"]
    )
    
    return {"final_status": status}

# Edges
def route_after_generation(state: AgentState):
    if state.get("error_feedback"):
        if state["retry_count"] >= 3:
            return "report"
        return "generate_and_execute"
    return "validate"

def route_after_validation(state: AgentState):
    if state.get("error_feedback") and state["retry_count"] < 3:
        return "generate_and_execute"
    return "report"

# Build Graph
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("profile_and_plan", node_profile_and_plan)
    workflow.add_node("generate_and_execute", node_generate_and_execute)
    workflow.add_node("validate", node_validate)
    workflow.add_node("report", node_report)
    
    workflow.set_entry_point("profile_and_plan")
    
    workflow.add_edge("profile_and_plan", "generate_and_execute")
    
    workflow.add_conditional_edges(
        "generate_and_execute",
        route_after_generation,
        {
            "validate": "validate",
            "generate_and_execute": "generate_and_execute",
            "report": "report"
        }
    )
    
    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "generate_and_execute": "generate_and_execute",
            "report": "report"
        }
    )
    
    workflow.add_edge("report", END)
    
    return workflow.compile()


@traceable(run_type="chain", name="Data Preprocessing Pipeline")
def run_pipeline(job_id: str, input_file: str, output_file: str, report_file: str) -> dict:
    """Top-level traceable entry point. Every CSV upload creates one LangSmith trace."""
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
