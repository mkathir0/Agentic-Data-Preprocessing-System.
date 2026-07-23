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
    """
    settings = get_settings()
    
    # Extract the model name if it starts with groq/
    model_name = settings.LLM_MODEL.replace("groq/", "") if settings.LLM_MODEL.startswith("groq/") else settings.LLM_MODEL
    
    # Initialize LangChain Groq model and bind structured output
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
    
    Generate a TransformationPlan detailing the steps to clean this dataset.
    For each action, specify the column it applies to (or None for global actions), the type of action 
    (e.g., 'drop_missing', 'fill_missing_median', 'drop_duplicates', 'cast_to_datetime'), and a brief explanation.
    """
    
    messages = [
        SystemMessage(content="You are a precise data engineering planner."),
        HumanMessage(content=prompt)
    ]
    
    plan = structured_llm.invoke(messages)
    return plan
