from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.agents.schemas import (
    ProfilerOutput,
    QualityOutput,
    SchemaOutput,
    TransformationPlan
)

def generate_transformation_plan(
    profiler_data: ProfilerOutput,
    quality_data: QualityOutput,
    schema_data: SchemaOutput
) -> TransformationPlan:
    """
    Uses an LLM to read the deterministic profiling data and output a structured cleaning plan.
    Each action must include: action_type, explanation, confidence (high/medium/low),
    assumption (any assumption made), and estimated_rows_affected_pct.
    """
    settings = get_settings()

    model_name = settings.LLM_MODEL.replace("groq/", "") if settings.LLM_MODEL.startswith("groq/") else settings.LLM_MODEL

    llm = ChatGroq(model=model_name, temperature=0, api_key=settings.GROQ_API_KEY)
    structured_llm = llm.with_structured_output(TransformationPlan)

    prompt = f"""
    You are an expert Data Engineer. You have been provided with deterministic data profiling,
    quality, and schema information for a messy dataset.

    Your task is to create a deterministic TransformationPlan to clean this dataset.
    Only suggest transformations that are strongly supported by the data anomalies provided.

    --- Profiler Data ---
    {profiler_data.model_dump_json(indent=2)}

    --- Quality Data (Anomalies) ---
    {quality_data.model_dump_json(indent=2)}

    --- Schema Inference ---
    {schema_data.model_dump_json(indent=2)}

    CRITICAL INSTRUCTIONS FOR ANOMALIES:
    You MUST propose a fix for ALL anomalies reported in the Quality Data. Do not ignore them!
    - For whitespace/format issues (e.g. leading/trailing spaces), propose a 'strip_whitespace' action (confidence: high). 
    - **IMPORTANT ORDERING**: You MUST apply whitespace stripping and formatting fixes BEFORE dropping exact duplicates. Trailing spaces can hide exact duplicates, so deduping must happen after whitespace fixes!
    - For cross-field business rule violations (e.g. ended_at < started_at) or range bounds (e.g. speed > 200), you MUST propose an action to fix them. If you can confidently fix them (e.g. 'drop_invalid_rows', 'swap_dates'), do so. If you cannot confidently fix them, you MUST propose a 'flag_for_review' action so they are explicitly escalated to a human.

    For EACH action you plan, you MUST provide:
    1. `action_type`: The transformation (e.g. 'drop_duplicates', 'fill_missing_median', 'cast_to_datetime', 'drop_column'). 
       **IMPORTANT**: If your confidence is "low", you MUST set the action_type to "flag_for_review" and do NOT suggest a data mutation.
    2. `column`: The column name, or null if global
    3. `explanation`: Why this transformation is needed based on the data
    4. `confidence`: 
       - "high" = well-supported by clear statistics (e.g. dropping exact duplicates, fixing clear dtype mismatch)
       - "medium" = reasonable heuristic but some uncertainty (e.g. median imputation on skewed data)
       - "low" = uncertain or may alter data meaning (e.g. filling an ID/PK column, imputing >20% of values). MUST use "flag_for_review".
    5. `assumption`: Any assumption you are making about the data that a human would need to know to audit this decision. 
       Examples: "assumed missing station_id = dockless GPS ride", "assumed negative prices are data entry errors not credits",
       "assumed duplicate ride_ids are true duplicates not re-used IDs across time periods".
       Write null if no assumption is being made.
       **IMPORTANT**: Do not blindly fill missing semantic/categorical data with 'Unknown'. If missingness has semantic meaning (e.g. missing bike station names imply a dockless ride), suggest creating a boolean indicator column (e.g. `is_dockless_start`, `is_dockless_end` - ensure names are unique to the source column!) and leaving the original column as null, rather than destroying the signal.
    6. `estimated_rows_affected_pct`: Your best estimate of what fraction of rows (0.0-1.0) this action will touch,
       based on the profiling data. E.g. if 5% of rows have nulls in this column, put 0.05.

    Also provide:
    - `dataset_summary`: A 1-2 sentence plain-English description of the dataset's current state
    - `global_assumptions`: A list of any high-level assumptions that apply across multiple steps
      (e.g. "This appears to be a bike-share trip dataset", "Duplicate IDs appear to be true duplicates not re-used keys")

    Generate a TransformationPlan now.
    """

    messages = [
        SystemMessage(content="You are a precise data engineering planner. Always fill in the confidence, assumption, and estimated_rows_affected_pct fields — never leave them missing."),
        HumanMessage(content=prompt)
    ]

    plan = structured_llm.invoke(messages)
    return plan
