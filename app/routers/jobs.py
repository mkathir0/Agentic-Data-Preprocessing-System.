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
            else:
                job.status = "COMPLETED"
                job.progress = 100.0
                job.cleaned_filepath = output_file
            db.commit()
            
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
    if job.status != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    phases = []

    # ── Phase 1: Profiling ─────────────────────────────────────────────────────
    pd = job.profiler_data or {}
    qb = job.quality_before or {}
    qb_anomalies = qb.get("anomalies", [])

    missing_cols = [
        a for a in qb_anomalies if a.get("anomaly_type") == "missing_values"
    ]
    dup_anomaly = next((a for a in qb_anomalies if a.get("anomaly_type") == "duplicate_rows"), None)
    mixed_type_cols = [a for a in qb_anomalies if a.get("anomaly_type") == "mixed_types"]
    high_sev = [a for a in qb_anomalies if a.get("severity") == "high"]
    med_sev = [a for a in qb_anomalies if a.get("severity") == "medium"]

    # Build column type breakdown
    col_types: dict = pd.get("column_types", {})
    numeric_cols = [c for c, t in col_types.items() if "Numeric" in t or "numeric" in t]
    text_cols = [c for c, t in col_types.items() if "Text" in t or "Categorical" in t or "text" in t]
    datetime_cols = [c for c, t in col_types.items() if "Date" in t or "date" in t]

    profiler_lines = []
    if pd:
        profiler_lines.append(f"Dataset has **{pd.get('num_rows', '?'):,} rows** and **{pd.get('num_columns', '?')} columns**")
        profiler_lines.append(f"Memory footprint: {round(pd.get('memory_usage_bytes', 0) / 1024, 1)} KB")
        if numeric_cols:
            profiler_lines.append(f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:6])}{' …' if len(numeric_cols) > 6 else ''}")
        if text_cols:
            profiler_lines.append(f"Text/categorical columns ({len(text_cols)}): {', '.join(text_cols[:6])}{' …' if len(text_cols) > 6 else ''}")
        if datetime_cols:
            profiler_lines.append(f"Date columns ({len(datetime_cols)}): {', '.join(datetime_cols)}")
    if qb_anomalies:
        profiler_lines.append(f"Pandera detected **{len(qb_anomalies)} anomalies**: {len(high_sev)} high-severity, {len(med_sev)} medium-severity")
    if dup_anomaly:
        profiler_lines.append(f"Duplicate rows found: **{dup_anomaly.get('affected_count', 0)}** ({round(dup_anomaly.get('affected_percentage', 0)*100, 1)}% of dataset)")
    if missing_cols:
        worst = sorted(missing_cols, key=lambda x: x.get('affected_percentage', 0), reverse=True)
        top = worst[:3]
        profiler_lines.append(
            "Top missing-value columns: " +
            ", ".join(f"'{a['column']}' ({round(a.get('affected_percentage',0)*100,1)}% null)" for a in top)
        )
    if mixed_type_cols:
        names = [a.get('column','?') for a in mixed_type_cols[:3]]
        profiler_lines.append(f"Mixed-type columns: {', '.join(names)} — values contain both strings and numbers")

    phases.append({
        "phase": "Phase 1 — Profiling & Quality Scan",
        "color": "cyan",
        "status": "completed",
        "lines": profiler_lines or ["No profiling data available for this job."],
    })

    # ── Phase 2: LLM Transformation Plan ──────────────────────────────────────
    plan = job.plan_data or {}
    actions = plan.get("actions", [])
    dataset_summary = plan.get("dataset_summary", "")

    plan_lines = []
    if dataset_summary:
        plan_lines.append(f"LLM assessment: \"*{dataset_summary}*\"")
    if actions:
        plan_lines.append(f"**{len(actions)} transformation steps** planned:")
        for i, action in enumerate(actions[:10], 1):
            col = f" on `{action['column']}`" if action.get('column') else " (whole dataset)"
            plan_lines.append(f"{i}. **{action['action_type']}**{col} — {action['explanation']}")
        if len(actions) > 10:
            plan_lines.append(f"… and {len(actions) - 10} more steps")
    else:
        plan_lines.append("No transformation plan available for this job.")

    phases.append({
        "phase": "Phase 2 — LLM Transformation Plan",
        "color": "purple",
        "status": "completed",
        "lines": plan_lines,
    })

    # ── Phase 3: Code Gen & Execution ─────────────────────────────────────────
    retry_count = int(job.retry_count or 0)
    exec_lines = []
    if retry_count == 0:
        exec_lines.append("LLM-generated Pandas cleaning script **executed successfully on the first attempt**")
    else:
        exec_lines.append(f"Initial script failed — LLM analysed the traceback and rewrote the code")
        exec_lines.append(f"Required **{retry_count} retry attempt{'s' if retry_count > 1 else ''}** before success")
    if job.cleaned_filepath:
        exec_lines.append(f"Output written to: `cleaned_{job.filename}`")

    phases.append({
        "phase": "Phase 3 — Code Generation & Execution",
        "color": "green" if retry_count == 0 else "amber",
        "status": "completed",
        "lines": exec_lines,
        "badge": "1 retry" if retry_count > 0 else "First pass",
    })

    # ── Phase 4: Post-clean Validation ────────────────────────────────────────
    qa = job.quality_after or {}
    qa_anomalies = qa.get("anomalies", [])
    qa_valid = qa.get("is_valid", True)
    qa_summary = qa.get("summary", "")

    val_lines = []
    if qa:
        if qa_valid:
            val_lines.append("Pandera post-clean validation **passed** — no high-severity anomalies remain")
        else:
            val_lines.append("Pandera post-clean validation detected remaining issues")
        if qa_summary:
            val_lines.append(qa_summary)
        # Compare before vs after
        before_count = len(qb_anomalies)
        after_count = len(qa_anomalies)
        if before_count > 0:
            resolved = before_count - after_count
            val_lines.append(f"Anomalies resolved: **{resolved} of {before_count}** ({round(resolved/before_count*100)}%)")
        remaining_high = [a for a in qa_anomalies if a.get("severity") == "high"]
        if remaining_high:
            val_lines.append(f"Remaining high-severity issues: {len(remaining_high)}")
    else:
        val_lines.append("Post-clean validation data not available for this job.")

    phases.append({
        "phase": "Phase 4 — Post-Clean Validation",
        "color": "green" if qa_valid else "red",
        "status": "completed" if qa_valid else "failed",
        "lines": val_lines,
        "badge": "✓ Passed" if qa_valid else "✗ Issues remain",
    })

    return {"job_id": job_id, "filename": job.filename, "phases": phases}



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

