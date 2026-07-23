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
    
    # Extract the model name if it starts with groq/
    model_name = settings.LLM_MODEL.replace("groq/", "") if settings.LLM_MODEL.startswith("groq/") else settings.LLM_MODEL
    
    # Initialize LangChain Groq model and bind structured output
    llm = ChatGroq(model=model_name, temperature=0, api_key=settings.GROQ_API_KEY)
    structured_llm = llm.with_structured_output(CodeGenerationOutput)
    
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
    
    Return ONLY valid, executable Python code in the 'code' field of the output.
    Ensure that any required imports (like pandas) are included.
    Do NOT include markdown backticks like ```python in the 'code' field.
    """
    
    messages = [
        SystemMessage(content="You are a precise data engineering coder."),
        HumanMessage(content=prompt)
    ]
    
    code_gen = structured_llm.invoke(messages)
    return code_gen
