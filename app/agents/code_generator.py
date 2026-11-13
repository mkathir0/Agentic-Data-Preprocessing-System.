from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_settings
from app.agents.schemas import TransformationPlan, CodeGenerationOutput

def generate_cleaning_code(plan: TransformationPlan, error_feedback: str = None, previous_code: str = None) -> CodeGenerationOutput:
    """
    Takes the structured TransformationPlan and generates executable Python pandas code.
    Can also take error_feedback and previous_code to fix mistakes.
    """
    settings = get_settings()

    # Graceful fallback if no GROQ_API_KEY is configured
    if not settings.GROQ_API_KEY:
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "",
            "def clean_data(df: pd.DataFrame) -> pd.DataFrame:",
            "    df = df.copy()",
            "    # 1. Strip whitespace from all string columns",
            "    for col in df.select_dtypes(include=['object', 'string']).columns:",
            "        df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan})",
            "",
            "    # 2. Drop exact duplicates",
            "    df = df.drop_duplicates().reset_index(drop=True)",
            "",
            "    # 3. Numeric bounds & sentinel string cleansing",
            "    for col in df.columns:",
            "        if any(term in col.lower() for term in ['amount', 'price', 'salary', 'fee', 'subtotal', 'total']):",
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
        try:
            llm = ChatGroq(
                model=cand,
                temperature=0,
                api_key=settings.GROQ_API_KEY,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            resp = llm.invoke(messages)
            content = resp.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            code_gen = CodeGenerationOutput.model_validate_json(content.strip())
            return code_gen
        except Exception as e:
            last_err = e
            try:
                llm_fallback = ChatGroq(model=cand, temperature=0, api_key=settings.GROQ_API_KEY)
                structured_llm = llm_fallback.with_structured_output(CodeGenerationOutput)
                code_gen = structured_llm.invoke(messages)
                return code_gen
            except Exception:
                continue
    raise last_err
