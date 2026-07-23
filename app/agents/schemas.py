from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ProfilerOutput(BaseModel):
    num_rows: int
    num_columns: int
    memory_usage_bytes: int
    missing_cells: int
    missing_cells_percentage: float
    duplicate_rows: int
    duplicate_rows_percentage: float
    column_types: Dict[str, str] # e.g. {"age": "Numeric", "name": "Text"}
    columns_summary: Dict[str, Any] # Detailed summary per column
    
class QualityAnomaly(BaseModel):
    column: Optional[str] = None
    anomaly_type: str # e.g. "missing_values", "mixed_types", "duplicate_rows"
    description: str
    severity: str # "high", "medium", "low"
    affected_count: Optional[int] = None
    affected_percentage: Optional[float] = None

class QualityOutput(BaseModel):
    is_valid: bool
    anomalies: List[QualityAnomaly]
    summary: str

class SchemaOutput(BaseModel):
    primary_keys: List[str] # Candidate primary keys (high cardinality)
    candidate_foreign_keys: List[str] # Columns that look like IDs
    categorical_columns: List[str] # Low cardinality string columns
    datetime_columns: List[str] # Columns inferred as datetimes
    inferred_types: Dict[str, str] # The inferred domain types (e.g., email, zip_code, url, etc.)

class TransformationAction(BaseModel):
    column: Optional[str] = Field(None, description="The specific column this action applies to, or null if it applies to the whole dataset")
    action_type: str = Field(..., description="The type of transformation (e.g. 'drop_duplicates', 'fill_missing', 'cast_type', 'drop_column')")
    explanation: str = Field(..., description="Explanation of why this action is being taken based on the data profile")

class TransformationPlan(BaseModel):
    dataset_summary: str = Field(..., description="A short summary of the dataset state based on the profiling data")
    actions: List[TransformationAction] = Field(..., description="The sequence of deterministic actions to clean the dataset")

class CodeGenerationOutput(BaseModel):
    code: str = Field(..., description="The executable Python pandas code that performs the actions. Do not include markdown code block syntax like ```python, just the raw code.")
    dependencies: List[str] = Field(..., description="List of pip packages required to run the code, e.g. ['pandas']")
