from pydantic import BaseModel, ConfigDict
from typing import Optional, Any, Dict
from datetime import datetime

class JobBase(BaseModel):
    filename: str

class JobUploadResponse(JobBase):
    id: str
    status: str
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class JobResponse(JobBase):
    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    progress: float
    current_step: Optional[str] = None
    
    original_filepath: Optional[str] = None
    cleaned_filepath: Optional[str] = None
    cleanup_script_path: Optional[str] = None
    cleaning_plan: Optional[Dict[str, Any]] = None
    profile_report_path: Optional[str] = None
    quality_report_path: Optional[str] = None
    validation_report_path: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
