import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, Text, JSON
from app.database import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, index=True)
    status = Column(String, default="PENDING")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    error_message = Column(Text, nullable=True)
    progress = Column(Float, default=0.0)
    current_step = Column(String, nullable=True)
    
    original_filepath = Column(String, nullable=True)
    cleaned_filepath = Column(String, nullable=True)
    cleanup_script_path = Column(String, nullable=True)
    cleaning_plan = Column(JSON, nullable=True)
    profile_report_path = Column(String, nullable=True)
    quality_report_path = Column(String, nullable=True)
    validation_report_path = Column(String, nullable=True)
