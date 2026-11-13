import json
import re
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.agents.schemas import (
    ProfilerOutput,
    QualityOutput,
    SchemaOutput,
    TransformationPlan,
    TransformationAction
)

logger = logging.getLogger(__name__)

def extract_json_from_response(text: str) -> dict:
    t = text.strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(t)
    except Exception:
        match = re.search(r"(\{.*\})", t, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise

def _generate_heuristic_plan(
    profiler_data: ProfilerOutput,
    quality_data: QualityOutput,
    schema_data: SchemaOutput
) -> TransformationPlan:
    actions = [
        TransformationAction(
            column=None,
            action_type="strip_whitespace",
            explanation="Trim untrimmed whitespace from string columns to prevent hidden duplicate keys.",
            confidence="high",
            estimated_rows_affected_pct=0.05
        ),
        TransformationAction(
            column=None,
            action_type="drop_duplicates",
            explanation=f"Drop {profiler_data.duplicate_rows} exact duplicate rows.",
            confidence="high",
            estimated_rows_affected_pct=profiler_data.duplicate_rows_percentage
        )
    ]
    for anom in quality_data.anomalies:
        if anom.anomaly_type == "missing_values" and anom.column:
            actions.append(TransformationAction(
                column=anom.column,
                action_type="fill_missing_median" if profiler_data.column_types.get(anom.column) == "Numeric" else "fill_missing_mode",
                explanation=f"Address missing values in '{anom.column}'.",
                confidence="medium",
                estimated_rows_affected_pct=anom.affected_percentage or 0.05
            ))
        elif anom.anomaly_type == "range_bound_violation" and anom.column:
            actions.append(TransformationAction(
                column=anom.column,
                action_type="clip_or_filter_bounds",
                explanation=f"Normalize out-of-bounds domain values in '{anom.column}'.",
                confidence="high",
                estimated_rows_affected_pct=anom.affected_percentage or 0.02
            ))
        elif anom.anomaly_type == "cross_field_violation":
            actions.append(TransformationAction(
                column=None,
                action_type="reconcile_cross_field",
                explanation="Reconcile temporal order and arithmetic identity violations.",
                confidence="high",
                estimated_rows_affected_pct=0.02
            ))
    return TransformationPlan(
        dataset_summary=f"Automated transformation plan addressing {len(quality_data.anomalies)} anomalies across {profiler_data.num_columns} columns.",
        actions=actions,
        global_assumptions=["Dataset cleaned using deterministic domain heuristics."]
    )

def generate_transformation_plan(
    profiler_data: ProfilerOutput,
    quality_data: QualityOutput,
    schema_data: SchemaOutput,
    user_instructions: str = None
) -> TransformationPlan:
    """
    Uses an LLM to read the deterministic profiling data and output a structured cleaning plan.
    Each action must include: action_type, explanation, confidence (high/medium/low),
    assumption (any assumption made), and estimated_rows_affected_pct.
    """
    settings = get_settings()

    # Graceful fallback if no GROQ_API_KEY is configured
    if not settings.GROQ_API_KEY:
        return _generate_heuristic_plan(profiler_data, quality_data, schema_data)

    model_name = settings.LLM_MODEL.replace("groq/", "") if settings.LLM_MODEL.startswith("groq/") else settings.LLM_MODEL
    candidates = list(dict.fromkeys([model_name, "openai/gpt-oss-120b", "openai/gpt-oss-20b"]))

    # Condense profiler and quality data into a compact summary to avoid rate limits (ITPM)
    lines_p = [
        f"Rows: {profiler_data.num_rows:,}, Columns: {profiler_data.num_columns}",
        f"Missing cells: {profiler_data.missing_cells:,} ({profiler_data.missing_cells_percentage:.1f}%)",
        f"Duplicate rows: {profiler_data.duplicate_rows:,} ({profiler_data.duplicate_rows_percentage:.1f}%)",
        "Columns:"
    ]
    for col, ctype in profiler_data.column_types.items():
        c_sum = profiler_data.columns_summary.get(col, {})
        n_miss = c_sum.get("n_missing", 0)
        lines_p.append(f"  - {col} ({ctype}): {n_miss} nulls")
    condensed_profiler = "\n".join(lines_p)

    lines_q = [f"Summary: {quality_data.summary}", "Detected Anomalies:"]
    for a in quality_data.anomalies:
        lines_q.append(f"  - [{a.severity.upper()}] {a.column or 'DATASET'}: {a.anomaly_type} - {a.description}")
    condensed_quality = "\n".join(lines_q)

    condensed_schema = (
        f"Primary Keys: {schema_data.primary_keys}\n"
        f"Categorical Columns: {schema_data.categorical_columns}\n"
        f"Datetime Columns: {schema_data.datetime_columns}\n"
        f"Inferred Types: {schema_data.inferred_types}"
    )

    instructions_block = ""
    if user_instructions and user_instructions.strip():
        instructions_block = f"""
    --- USER CLEANING GOALS & REQUIREMENTS ---
    The user explicitly requested the following cleaning requirements:
    "{user_instructions.strip()}"
    You MUST honor and incorporate these user-specified instructions into the transformation actions.
"""

    prompt = f"""
    You are an expert Data Engineer. You have been provided with deterministic data profiling,
    quality, and schema information for a messy dataset.

    Your task is to create a deterministic TransformationPlan to clean this dataset.
    Only suggest transformations that are strongly supported by the data anomalies provided.
{instructions_block}
    --- Profiler Summary ---
    {condensed_profiler}

    --- Quality Anomalies ---
    {condensed_quality}

    --- Schema Inference ---
    {condensed_schema}

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

    CRITICAL: You MUST respond ONLY with a valid JSON object matching this schema:
    {{
      "dataset_summary": "1-2 sentence overview of dataset issues",
      "global_assumptions": ["assumption 1", "assumption 2"],
      "actions": [
        {{
          "action_type": "strip_whitespace",
          "column": "column_name or null",
          "explanation": "why this action is needed",
          "confidence": "high",
          "assumption": "assumption text or null",
          "estimated_rows_affected_pct": 0.05
        }}
      ]
    }}
    """

    messages = [
        SystemMessage(content="You are a precise data engineering planner. Always output valid JSON matching the requested schema. Fill in confidence, assumption, and estimated_rows_affected_pct fields."),
        HumanMessage(content=prompt)
    ]

    last_err = None
    for cand in candidates:
        # Strategy A: Plain text invocation (No response_format) - avoids Groq gateway 400 json_validate_failed
        try:
            llm = ChatGroq(model=cand, temperature=0, api_key=settings.GROQ_API_KEY)
            resp = llm.invoke(messages)
            parsed = extract_json_from_response(resp.content)
            plan = TransformationPlan.model_validate(parsed)
            return plan
        except Exception as e:
            last_err = e

        # Strategy B: JSON mode with response_format
        try:
            llm = ChatGroq(
                model=cand,
                temperature=0,
                api_key=settings.GROQ_API_KEY,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            resp = llm.invoke(messages)
            parsed = extract_json_from_response(resp.content)
            plan = TransformationPlan.model_validate(parsed)
            return plan
        except Exception as e:
            last_err = e

        # Strategy C: with_structured_output
        try:
            llm_fallback = ChatGroq(model=cand, temperature=0, api_key=settings.GROQ_API_KEY)
            structured_llm = llm_fallback.with_structured_output(TransformationPlan)
            plan = structured_llm.invoke(messages)
            return plan
        except Exception as e:
            last_err = e

    logger.warning(f"All LLM planning candidates failed ({last_err}). Falling back to heuristic plan.")
    return _generate_heuristic_plan(profiler_data, quality_data, schema_data)
