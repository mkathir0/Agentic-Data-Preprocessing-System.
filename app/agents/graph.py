import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from app.agents.profiler import run_profiler
from app.agents.quality import run_quality_check
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
    
    # Status
    final_status: str

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
    # If code crashed during execution, don't try to validate the non-existent output
    if state.get("error_feedback"):
        return state
        
    quality = run_quality_check(state["output_file"])
    
    if not quality.is_valid:
        anomalies_str = str([a.model_dump() for a in quality.anomalies if a.severity == "high"])
        return {
            "error_feedback": f"Validation failed. Remaining high-severity anomalies on the cleaned data: {anomalies_str}",
            "retry_count": state["retry_count"] + 1,
            "quality_summary": quality.summary
        }
    else:
        return {
            "error_feedback": None,
            "quality_summary": quality.summary
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
def run_pipeline(job_id: str, input_file: str, output_file: str, report_file: str) -> str:
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
        "final_status": "PROCESSING",
    }
    result = graph.invoke(initial_state)
    return result.get("final_status", "COMPLETED")
