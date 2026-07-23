import os
import shutil
import asyncio
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Job
from app.schemas import JobResponse, JobUploadResponse
from app.config import settings

from app.security import verify_api_key

router = APIRouter(
    prefix="/api/jobs", 
    tags=["jobs"],
    dependencies=[Depends(verify_api_key)]
)

from app.agents.graph import run_pipeline

def process_job(job_id: str, db: Session):
    def update_status(prog: float, status: str = "PROCESSING", err: str = None, cleaned_file: str = None):
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.progress = prog
            job.status = status
            if err:
                job.error_message = err
            if cleaned_file:
                job.cleaned_filepath = cleaned_file
            db.commit()

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return
    
    input_file = job.original_filepath
    output_file = os.path.join(settings.OUTPUT_DIR, f"cleaned_{job.filename}")
    report_file = os.path.join(settings.OUTPUT_DIR, f"report_{job.filename.replace('.csv', '.md')}")
    
    update_status(10.0, "PROCESSING")
    
    # Run the pipeline — this is a @traceable function, so every job
    # automatically appears as a named run in LangSmith.
    try:
        final_status = run_pipeline(
            job_id=job_id,
            input_file=input_file,
            output_file=output_file,
            report_file=report_file,
        )
        
        if final_status == "FAILED":
            update_status(100.0, status="FAILED", err="Validation or execution failed. Check LangSmith for trace.")
        else:
            update_status(100.0, status="COMPLETED", cleaned_file=output_file)
            
    except Exception as e:
        update_status(100.0, status="FAILED", err=str(e))


@router.post("/upload", response_model=JobUploadResponse)
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    
    # Save the file (Size limits removed per request)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create job in database
    new_job = Job(
        filename=file.filename,
        original_filepath=file_path,
        status="PENDING",
        current_step="PENDING",
        progress=0.0
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Start execution task
    background_tasks.add_task(process_job, new_job.id, db)

    return new_job

@router.get("/", response_model=List[JobResponse])
def get_all_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).all()
    return jobs

@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

from fastapi.responses import FileResponse

@router.get("/{job_id}/download")
def download_cleaned_csv(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
        
    if not job.cleaned_filepath or not os.path.exists(job.cleaned_filepath):
        raise HTTPException(status_code=404, detail="Cleaned file not found")
        
    return FileResponse(
        path=job.cleaned_filepath, 
        filename=f"cleaned_{job.filename}", 
        media_type='text/csv'
    )

@router.get("/{job_id}/report")
def get_report(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.status != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
        
    report_file = os.path.join(settings.OUTPUT_DIR, f"report_{job.filename.replace('.csv', '.md')}")
    if not os.path.exists(report_file):
        raise HTTPException(status_code=404, detail="Report not found")
        
    # Return as raw markdown text so the frontend can parse it
    return FileResponse(
        path=report_file,
        filename=os.path.basename(report_file),
        media_type='text/markdown'
    )
