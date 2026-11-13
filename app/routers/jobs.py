import os
import shutil
import asyncio
import json
import logging
import uuid
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.job_artifacts import artifact_paths
from app.models import Job
from app.job_queue import enqueue_job
from app.schemas import JobResponse, JobUploadResponse
from app.config import settings

from app.security import verify_api_key

router = APIRouter(
    prefix="/api/jobs", 
    tags=["jobs"],
    dependencies=[Depends(verify_api_key)]
)

from app.agents.graph import run_pipeline

def process_job(job_id: str, db: Session = None):
    if db is not None:
        return _process_job(job_id, db)
    with SessionLocal() as worker_db:
        return _process_job(job_id, worker_db)


def _process_job(job_id: str, db: Session):
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
    artifacts = artifact_paths(settings.OUTPUT_DIR, job.id, job.filename)
    artifacts['directory'].mkdir(parents=True, exist_ok=True)
    output_file = str(artifacts['cleaned'])
    report_file = str(artifacts['report'])
    job.cleaned_filepath = None
    job.profile_report_path = None
    job.cleanup_script_path = None
    job.quality_report_path = None
    job.validation_report_path = None
    
    update_status(10.0, "PROCESSING")
    
    try:
        result = run_pipeline(
            job_id=job_id,
            input_file=input_file,
            output_file=output_file,
            report_file=report_file,
            user_instructions=job.user_instructions,
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

            # Persist generated Python cleaning code
            current_code = result.get("current_code")
            if current_code:
                script_path = str(artifacts['script'])
                Path(script_path).write_text(current_code, encoding="utf-8")
                job.cleanup_script_path = script_path

            report = result.get("validation_report") or {}
            has_output = os.path.isfile(output_file) and os.path.getsize(output_file) > 0
            integrity_passed = report.get("integrity_passed") is True
            validated = report.get("overall_passed") is True and integrity_passed
            job.progress = 100.0
            job.cleaned_filepath = output_file if has_output and integrity_passed else None
            job.profile_report_path = report_file if os.path.isfile(report_file) else None
            job.quality_report_path = str(artifacts['suite']) if os.path.isfile(artifacts['suite']) else None
            job.validation_report_path = str(artifacts['validation']) if os.path.isfile(artifacts['validation']) else None
            if final_status == "COMPLETED" and has_output and validated:
                job.status = "COMPLETED"
                job.current_step = "COMPLETED"
                job.error_message = None
            else:
                job.status = "FAILED"
                if has_output and integrity_passed:
                    job.current_step = "REVIEW_FLAGGED"
                    job.error_message = "Records were preserved, but validation or planned actions require human review. Output is not approved for production."
                else:
                    job.current_step = "VALIDATION_FAILED" if has_output else "EXECUTION_FAILED"
                    job.error_message = "Output failed integrity checks or execution did not produce a verified dataset."
            db.commit()
            
    except Exception as e:
        db.rollback()
        failed_job = db.query(Job).filter(Job.id == job_id).first()
        if failed_job:
            failed_job.cleaned_filepath = None
            failed_job.current_step = "EXECUTION_FAILED"
        update_status(100.0, status="FAILED", err=str(e))


@router.post("/upload", response_model=List[JobUploadResponse])
async def upload_files(
    files: List[UploadFile] = File(...),
    instructions: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    new_jobs = []
    
    for file in files:
        name = file.filename or ""
        if not name or name in {".", ".."} or any(c in name for c in ("/", "\\", "\x00")):
            raise HTTPException(status_code=400, detail="Use a plain filename without directory components.")
        if Path(name).suffix.lower() not in {".csv", ".tsv", ".json", ".xlsx", ".xls", ".parquet"}:
            raise HTTPException(status_code=400, detail="Unsupported dataset format.")

    for file in files:
        job_id = str(uuid.uuid4())
        upload_dir = Path(settings.UPLOAD_DIR) / job_id
        upload_dir.mkdir(parents=True, exist_ok=False)
        file_path = str(upload_dir / file.filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Create job in database
        new_job = Job(
            id=job_id,
            filename=file.filename,
            original_filepath=file_path,
            status="PENDING",
            current_step="PENDING",
            progress=0.0,
            user_instructions=instructions
        )
        db.add(new_job)
        enqueue_job(db, job_id)
        db.commit()
        db.refresh(new_job)
        new_jobs.append(new_job)

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

@router.get("/{job_id}/raw")
def get_raw_csv(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job or not job.original_filepath or not os.path.exists(job.original_filepath):
        raise HTTPException(status_code=404, detail="Original dataset not found")
    return FileResponse(
        path=job.original_filepath,
        filename=job.filename,
        media_type=mimetypes.guess_type(job.filename)[0] or 'application/octet-stream'
    )

@router.get("/{job_id}/download")
def download_cleaned_csv(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in ["COMPLETED", "FAILED"]:
        raise HTTPException(status_code=400, detail="Job is not completed yet")
        
    if not job.cleaned_filepath or not os.path.exists(job.cleaned_filepath):
        raise HTTPException(status_code=404, detail="Cleaned file not found")
    if not (job.validation_report or {}).get("integrity_passed"):
        raise HTTPException(status_code=409, detail="This output has not passed integrity verification. Re-run the dataset before using it.")
        
    return FileResponse(
        path=job.cleaned_filepath, 
        filename=f"{'cleaned' if job.status == 'COMPLETED' else 'review_candidate'}_{Path(job.filename).stem}.csv",
        media_type='text/csv'
    )

@router.get("/{job_id}/report")
def get_report(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ["COMPLETED", "FAILED"]:
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    report_file = job.profile_report_path
    if not report_file or not os.path.isfile(report_file):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path=report_file, filename=os.path.basename(report_file), media_type='text/markdown')


@router.get("/{job_id}/quality-suite")
def get_quality_suite(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    suite_path = job.quality_report_path
    if not suite_path or not os.path.isfile(suite_path):
        raise HTTPException(status_code=404, detail="Great Expectations suite not found")
    return FileResponse(path=suite_path, filename=os.path.basename(suite_path), media_type="application/json")


@router.get("/{job_id}/validation-report")
def get_validation_report(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    report_path = job.validation_report_path
    if not report_path or not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="Validation report not found")
    return FileResponse(path=report_path, filename=os.path.basename(report_path), media_type="application/json")

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

    cleaning_code = None
    if job.cleanup_script_path and os.path.exists(job.cleanup_script_path):
        try:
            with open(job.cleanup_script_path, "r") as sf:
                cleaning_code = sf.read()
        except Exception:
            pass

    return {
        "job_id": job.id,
        "filename": job.filename,
        "user_instructions": job.user_instructions,
        "profiler_data": job.profiler_data,
        "plan_data": job.plan_data,
        "validation_report": job.validation_report,
        "retry_count": job.retry_count,
        "cleaning_code": cleaning_code
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
    elif "validation" in name_lower or "validate" in name_lower:
        return "ValidationEngine"
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
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="LangSmith is not configured: API key is missing.")

    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
    project_name = os.environ.get("LANGSMITH_PROJECT", "agentic-data-preprocessing-pipeline")

    try:
        from app.langsmith_queries import fetch_job_runs
        runs = await fetch_job_runs(job_id, job.created_at, endpoint, api_key, project_name)
        if not runs:
            return []

        events = []
        for r in sorted(runs, key=lambda run: (run.start_time, str(run.id))):
            agent_name = map_run_to_agent_name(r.name or "")
            dur_ms = int((r.end_time - r.start_time).total_seconds() * 1000) if (r.end_time and r.start_time) else None
            if r.error:
                msg = f"Error in {agent_name}: {r.error}"
                level = "error"
                status = "error"
            elif r.end_time is None:
                msg = f"{agent_name}: completion has not yet been received by LangSmith"
                level = "warning"
                status = "pending"
            else:
                msg = f"{agent_name} executed in {dur_ms}ms"
                level = "info"
                status = "completed"

            events.append({
                "id": str(r.id),
                "trace_id": str(r.trace_id),
                "parent_run_id": str(r.parent_run_ids[-1]) if r.parent_run_ids else None,
                "source": "langsmith",
                "status": status,
                "agent": agent_name,
                "level": level,
                "message": msg,
                "timestamp": r.start_time.isoformat() if r.start_time else "",
                "duration_ms": dur_ms,
                "tags": r.tags or [],
                "inputs": r.inputs or {},
                "outputs": r.outputs or {}
            })
        return events
    except Exception as exc:
        logging.getLogger(__name__).exception("LangSmith query failed for job %s", job_id)
        raise HTTPException(status_code=502, detail="Could not retrieve LangSmith traces. Please retry.") from exc


@router.get("/{job_id}/summary")
@router.get("/{job_id}/phase-summary")
def get_job_summary(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    report_file = job.profile_report_path
    report_exists = bool(report_file and os.path.isfile(report_file))

    phases = [
        {
            "phase": "Profiling & Analysis",
            "agent": "ProfilerAgent / QualityAgent / SchemaAgent",
            "status": "COMPLETED" if job.profiler_data is not None else ("PROCESSING" if job.status == "PROCESSING" else "PENDING"),
            "details": "Dataset profiling, Pandera data quality checks, and schema inference."
        },
        {
            "phase": "Transformation Planning",
            "agent": "PlannerAgent",
            "status": "COMPLETED" if job.status in ["COMPLETED", "FAILED"] and job.plan_data else ("PROCESSING" if job.status == "PROCESSING" else "PENDING"),
            "details": str(job.plan_data) if job.plan_data else "Generates deterministic transformation plan from profiling findings."
        },
        {
            "phase": "Code Generation & Execution",
            "agent": "CodeGeneratorAgent / ExecutorAgent",
            "status": "COMPLETED" if job.status == "COMPLETED" else ("FAILED" if job.status == "FAILED" else "PENDING"),
            "details": "Generates a cleaning script and executes it in a subprocess with timeout protection and automatic retry."
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

