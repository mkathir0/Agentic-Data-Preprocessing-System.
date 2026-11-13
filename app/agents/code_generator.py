import json
import re
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

from app.config import get_settings
from app.agents.schemas import TransformationPlan, CodeGenerationOutput

logger = logging.getLogger(__name__)

def extract_code_from_response(text: str) -> CodeGenerationOutput:
    t = text.strip()
    
    # Try 1: JSON parsing
    try:
        cand_t = t
        if "```json" in cand_t:
            cand_t = cand_t.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cand_t and "{" in cand_t:
            cand_t = cand_t.split("```", 1)[1].split("```", 1)[0].strip()
        
        try:
            d = json.loads(cand_t)
        except Exception:
            m = re.search(r"(\{.*\})", cand_t, re.DOTALL)
            if m:
                d = json.loads(m.group(1))
            else:
                raise
        return CodeGenerationOutput.model_validate(d)
    except Exception:
        pass

    # Try 2: Direct Python code extraction if LLM emitted python code directly
    if "def clean_data" in t:
        code_t = t
        if "```python" in code_t:
            code_t = code_t.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in code_t:
            code_t = code_t.split("```", 1)[1].split("```", 1)[0].strip()
        return CodeGenerationOutput(
            code=code_t,
            dependencies=["pandas", "numpy"]
        )

    raise ValueError(f"Could not extract Python code or JSON from response: {text[:200]}")

def _generate_heuristic_code(plan: TransformationPlan = None) -> CodeGenerationOutput:
    code_lines = [
        "import pandas as pd",
        "import numpy as np",
        "",
        "def clean_data(df: pd.DataFrame) -> pd.DataFrame:",
        "    df = df.copy()",
        "    # 1. Strip whitespace from all string columns",
        "    for col in df.select_dtypes(include=['object', 'string']).columns:",
        "        df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})",
        "",
        "    # 2. Drop exact duplicates",
        "    df = df.drop_duplicates().reset_index(drop=True)",
        "",
        "    # 3. Numeric bounds & sentinel string cleansing",
        "    for col in df.columns:",
        "        if any(term in col.lower() for term in ['amount', 'price', 'salary', 'fee', 'subtotal', 'total', 'tax']):",
        "            df[col] = pd.to_numeric(df[col].replace(['N/A', '-999', 'NULL', 'unknown'], np.nan), errors='coerce')",
        "            df[col] = df[col].abs()",
        "        elif 'age' in col.lower():",
        "            df[col] = pd.to_numeric(df[col], errors='coerce').clip(0, 120)",
        "",
        "    # 4. Fill missing numeric values with column median",
        "    for col in df.select_dtypes(include=[np.number]).columns:",
        "        if df[col].isnull().any():",
        "            df[col] = df[col].fillna(df[col].median())",
        "",
        "    return df"
    ]
    return CodeGenerationOutput(
        code="\n".join(code_lines),
        dependencies=["pandas", "numpy"]
    )

@traceable(name="Code Generator Agent (LLM)", run_type="chain")
def generate_cleaning_code(plan: TransformationPlan, error_feedback: str = None, previous_code: str = None) -> CodeGenerationOutput:
    """
    Takes the structured TransformationPlan and generates executable Python pandas code.
    Can also take error_feedback and previous_code to fix mistakes.
    """
    settings = get_settings()

    # Graceful fallback if no GROQ_API_KEY is configured
    if not settings.GROQ_API_KEY:
        return _generate_heuristic_code(plan)

    # Extract the model name if it starts with groq/
    model_name = settings.LLM_MODEL.replace("groq/", "") if settings.LLM_MODEL.startswith("groq/") else settings.LLM_MODEL
    candidates = list(dict.fromkeys([model_name, "openai/gpt-oss-120b", "openai/gpt-oss-20b"]))
    
    prompt = f"""
    You are an expert Data Engineer writing production Python code.
    
    You have been provided with a TransformationPlan:
    {plan.model_dump_json(indent=2)}
    """
    
    if error_feedback and previous_code:
        prompt += f"""
    HOWEVER, your previous code failed with the following error/validation issue:
    {error_feedback}
    
    Here was your previous code:
    {previous_code}
    
    Please fix the code to resolve these issues.
    """
    
    prompt += """
    Your task is to write a standalone Python script that executes this plan.
    Assume the data is passed as a pandas DataFrame named `df`.
    You should write a function called `clean_data(df: pd.DataFrame) -> pd.DataFrame`
    that implements the actions described in the plan.
    
    CRITICAL: If an action in the TransformationPlan is 'flag_for_review' or has 'low' confidence,
    **DO NOT** write pandas code to impute, drop, or alter that data. You MUST preserve the original
    values (including NaNs) for human review. You may add a python comment about it, but do not mutate the df for that step.
    
    Return ONLY valid, executable Python code in the 'code' field of the output.
    Ensure that any required imports (like pandas) are included.
    Do NOT include markdown backticks like ```python in the 'code' field.

    CRITICAL: You MUST respond with a valid JSON object matching this schema:
    {{
      "code": "import pandas as pd\\n\\ndef clean_data(df: pd.DataFrame) -> pd.DataFrame:\\n    # implementation\\n    return df",
      "dependencies": ["pandas", "numpy"]
    }}
    """
    
    messages = [
        SystemMessage(content="You are a precise data engineering coder. Always output valid JSON with 'code' and 'dependencies' fields."),
        HumanMessage(content=prompt)
    ]
    
    last_err = None
    for cand in candidates:
        # Strategy A: Plain text invocation (No response_format) - avoids Groq gateway 400 json_validate_failed
        try:
            llm = ChatGroq(model=cand, temperature=0, api_key=settings.GROQ_API_KEY)
            resp = llm.invoke(messages)
            return extract_code_from_response(resp.content)
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
            return extract_code_from_response(resp.content)
        except Exception as e:
            last_err = e

        # Strategy C: with_structured_output
        try:
            llm_fallback = ChatGroq(model=cand, temperature=0, api_key=settings.GROQ_API_KEY)
            structured_llm = llm_fallback.with_structured_output(CodeGenerationOutput)
            code_gen = structured_llm.invoke(messages)
            return code_gen
        except Exception as e:
            last_err = e

    logger.warning(f"All LLM code generation candidates failed ({last_err}). Falling back to heuristic code.")
    return _generate_heuristic_code(plan)
