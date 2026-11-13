import os
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx

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
    
    try:
        result = run_pipeline(
            job_id=job_id,
            input_file=input_file,
            output_file=output_file,
            report_file=report_file,
        )
        
        final_status = result.get("final_status", "FAILED")

        # Persist all phase data back to the DB row
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            # profiler_data is a ProfilerOutput pydantic object — convert to dict
            pd_raw = result.get("profiler_data")
            if pd_raw is not None:
                job.profiler_data = pd_raw.model_dump() if hasattr(pd_raw, "model_dump") else pd_raw

            # quality_data is already model_dump()'d in node_profile_and_plan
            job.quality_before = result.get("quality_data")

            # quality_after is already model_dump()'d in node_validate
            job.quality_after = result.get("quality_after")
            
            # validation_report is already model_dump()'d in node_validate
            job.validation_report = result.get("validation_report")

            # plan_json is a JSON string — parse it back to dict for storage
            plan_json_str = result.get("plan_json")
            if plan_json_str:
                import json
                try:
                    job.plan_data = json.loads(plan_json_str)
                except Exception:
                    job.plan_data = None

            job.retry_count = result.get("retry_count", 0)

            if final_status == "FAILED":
                job.status = "FAILED"
                job.progress = 100.0
                job.error_message = "Validation or execution failed after max retries."
                if os.path.exists(output_file):
                    job.cleaned_filepath = output_file
            else:
                job.status = "COMPLETED"
                job.progress = 100.0
                job.error_message = None
                job.cleaned_filepath = output_file
            db.commit()
            
    except Exception as e:
        update_status(100.0, status="FAILED", err=str(e))


@router.post("/upload", response_model=List[JobUploadResponse])
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    new_jobs = []
    
    for file in files:
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        
        # Save the file
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
        new_jobs.append(new_job)

        # Start execution task
        background_tasks.add_task(process_job, new_job.id, db)

    return new_jobs

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
    
    if job.status not in ["COMPLETED", "FAILED"]:
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
    if job.status not in ["COMPLETED", "FAILED"]:
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    report_file = os.path.join(settings.OUTPUT_DIR, f"report_{job.filename.replace('.csv', '.md')}")
    if not os.path.exists(report_file):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path=report_file, filename=os.path.basename(report_file), media_type='text/markdown')

@router.get("/{job_id}/insights")
def get_insights(job_id: str, db: Session = Depends(get_db)):
    """
    Returns structured, factual per-phase insights from the actual pipeline run.
    Every number comes directly from what the agents computed — nothing is invented.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ["COMPLETED", "FAILED"]:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    return {
        "job_id": job.id,
        "filename": job.filename,
        "profiler_data": job.profiler_data,
        "plan_data": job.plan_data,
        "validation_report": job.validation_report,
        "retry_count": job.retry_count
    }



def map_run_to_agent_name(name: str) -> str:
    if not name:
        return "UnknownAgent"
    name_lower = name.lower()
    if "run_profiler" in name_lower or "profileragent" in name_lower:
        return "ProfilerAgent"
    elif "run_quality_check" in name_lower or "qualityagent" in name_lower:
        return "QualityAgent"
    elif "generate_transformation_plan" in name_lower or "planneragent" in name_lower:
        return "PlannerAgent"
    elif "generate_cleaning_code" in name_lower or "codegeneratoragent" in name_lower:
        return "CodeGeneratorAgent"
    elif "execute_cleaning_code" in name_lower or "executoragent" in name_lower:
        return "ExecutorAgent"
    elif "generate_report" in name_lower or "reporteragent" in name_lower:
        return "ReporterAgent"
    elif "chatgroq" in name_lower or "groq" in name_lower or "llm" in name_lower:
        return "LLM (Groq)"
    elif "schema" in name_lower:
        return "SchemaAgent"
    elif "data preprocessing pipeline" in name_lower or "pipeline" in name_lower:
        return "Pipeline"
    return name


def calculate_duration_ms(run: dict) -> int:
    start_time = run.get("start_time")
    end_time = run.get("end_time")
    if not start_time or not end_time:
        return 0
    try:
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            return int((end_time - start_time) * 1000)
        from datetime import datetime
        s_dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
        e_dt = datetime.fromisoformat(str(end_time).replace("Z", "+00:00"))
        return int((e_dt - s_dt).total_seconds() * 1000)
    except Exception:
        return 0


def build_run_message(agent_name: str, run: dict, duration_ms: int) -> str:
    error = run.get("error")
    if error:
        return f"Error in {agent_name}: {error}"

    if agent_name == "PlannerAgent":
        outputs = run.get("outputs") or {}
        outputs_str = str(outputs)
        if len(outputs_str) > 150:
            outputs_str = outputs_str[:150] + "..."
        return f"PlannerAgent plan: {outputs_str}"

    return f"{agent_name} completed successfully in {duration_ms}ms"


@router.get("/{job_id}/langsmith-traces")
async def get_langsmith_traces(job_id: str, db: Session = Depends(get_db)):
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        return []

    project_name = os.environ.get("LANGSMITH_PROJECT", "Agentic Data-Preprocessing Pipeline")
    headers = {"x-api-key": api_key}
    params = {
        "project_name": project_name,
        "run_type": "chain",
        "limit": 50,
        "is_root": "true"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://api.smith.langchain.com/api/v1/runs", headers=headers, params=params)
            if resp.status_code != 200:
                return []

            runs_data = resp.json()
            runs = runs_data if isinstance(runs_data, list) else runs_data.get("runs", [])

            matching_runs = [r for r in runs if r.get("name") == "Data Preprocessing Pipeline"]
            if not matching_runs:
                return []

            # Match target root run for this job_id, or pick the most recent root run
            target_run = None
            for r in matching_runs:
                r_inputs = r.get("inputs") or {}
                if str(r_inputs.get("job_id")) == str(job_id):
                    target_run = r
                    break

            if not target_run:
                target_run = matching_runs[0]

            target_run_id = target_run.get("id")

            # Fetch child runs
            child_resp = await client.get(
                "https://api.smith.langchain.com/api/v1/runs",
                headers=headers,
                params={"parent_run": target_run_id}
            )

            child_runs = []
            if child_resp.status_code == 200:
                c_data = child_resp.json()
                child_runs = c_data if isinstance(c_data, list) else c_data.get("runs", [])

            all_runs = [target_run] + child_runs

            events = []
            for r in all_runs:
                name = r.get("name", "")
                agent_name = map_run_to_agent_name(name)
                dur_ms = calculate_duration_ms(r)
                msg = build_run_message(agent_name, r, dur_ms)

                err = r.get("error")
                if err or r.get("status") == "error":
                    level = "error"
                elif r.get("extra", {}).get("retry"):
                    level = "warning"
                else:
                    level = "info"

                events.append({
                    "id": str(r.get("id", "")),
                    "agent": agent_name,
                    "level": level,
                    "message": msg,
                    "timestamp": str(r.get("start_time") or r.get("created_at") or ""),
                    "duration_ms": dur_ms,
                    "tags": r.get("tags") or [],
                    "inputs": r.get("inputs") or {},
                    "outputs": r.get("outputs") or {}
                })

            return events

    except Exception:
        return []


@router.get("/{job_id}/summary")
@router.get("/{job_id}/phase-summary")
def get_job_summary(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    report_file = os.path.join(settings.OUTPUT_DIR, f"report_{job.filename.replace('.csv', '.md')}")
    report_exists = os.path.exists(report_file)

    phases = [
        {
            "phase": "Profiling & Analysis",
            "agent": "ProfilerAgent / QualityAgent / SchemaAgent",
            "status": "COMPLETED" if job.status in ["PROCESSING", "COMPLETED", "FAILED"] else "PENDING",
            "details": "Dataset profiling, Pandera data quality checks, and schema inference."
        },
        {
            "phase": "Transformation Planning",
            "agent": "PlannerAgent",
            "status": "COMPLETED" if job.status in ["COMPLETED", "FAILED"] and job.cleaning_plan else ("PROCESSING" if job.status == "PROCESSING" else "PENDING"),
            "details": job.cleaning_plan if job.cleaning_plan else "Generates deterministic transformation plan from profiling findings."
        },
        {
            "phase": "Code Generation & Execution",
            "agent": "CodeGeneratorAgent / ExecutorAgent",
            "status": "COMPLETED" if job.status == "COMPLETED" else ("FAILED" if job.status == "FAILED" else "PENDING"),
            "details": "Generates Pandas cleaning script and executes in isolated environment with automatic retry."
        },
        {
            "phase": "Validation & Verification",
            "agent": "QualityAgent",
            "status": "COMPLETED" if job.status == "COMPLETED" else ("FAILED" if job.status == "FAILED" else "PENDING"),
            "details": "Verifies cleaned dataset against quality criteria to ensure anomaly resolution."
        },
        {
            "phase": "Engineering Report",
            "agent": "ReporterAgent",
            "status": "COMPLETED" if report_exists else "PENDING",
            "details": "Compiles comprehensive engineering audit log and markdown summary report."
        }
    ]

    return {
        "job_id": job.id,
        "filename": job.filename,
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "error_message": job.error_message,
        "original_filepath": job.original_filepath,
        "cleaned_filepath": job.cleaned_filepath,
        "report_available": report_exists,
        "phases": phases
    }

