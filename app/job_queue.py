"""Database-backed job queue with leases and crash recovery."""
from datetime import datetime, timedelta
import logging
import socket
import time
import uuid

from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError

from app.database import SessionLocal
from app.models import Job, JobTask

logger = logging.getLogger(__name__)


def enqueue_job(db, job_id: str) -> None:
    if not db.query(JobTask).filter(JobTask.job_id == job_id).first():
        db.add(JobTask(job_id=job_id, status="QUEUED"))


def recover_jobs_without_tasks() -> int:
    """Adopt pending jobs created before the durable queue was introduced."""
    with SessionLocal() as db:
        terminal_job_ids = select(Job.id).where(Job.status.in_(["COMPLETED", "FAILED"]))
        db.query(JobTask).filter(
            JobTask.job_id.in_(terminal_job_ids), JobTask.status != "DONE"
        ).update(
            {JobTask.status: "DONE", JobTask.lease_expires_at: None, JobTask.updated_at: datetime.utcnow()},
            synchronize_session=False,
        )
        jobs = db.query(Job).outerjoin(JobTask, Job.id == JobTask.job_id).filter(
            JobTask.job_id.is_(None), Job.status.in_(["PENDING", "PROCESSING"])
        ).all()
        for job in jobs:
            # A PROCESSING job with no task record belonged to an in-process
            # BackgroundTask and was lost when its API process stopped.
            if job.status == "PROCESSING":
                job.status = "PENDING"
                job.current_step = "PENDING"
                job.progress = 0
            db.add(JobTask(job_id=job.id, status="QUEUED"))
        try:
            db.commit()
            return len(jobs)
        except IntegrityError:
            # Another worker may have adopted the same legacy row at startup.
            db.rollback()
            return 0


def claim_next_job(worker_id: str, lease_seconds: int = 90) -> str | None:
    """Claim one queued or expired task using a conditional database update."""
    now = datetime.utcnow()
    expires = now + timedelta(seconds=lease_seconds)
    with SessionLocal() as db:
        active_job_ids = select(Job.id).where(Job.status.in_(["PENDING", "PROCESSING"]))
        candidates = db.query(JobTask).filter(or_(
            JobTask.status == "QUEUED",
            and_(JobTask.status == "RUNNING", JobTask.lease_expires_at <= now),
        ), JobTask.job_id.in_(active_job_ids)).order_by(JobTask.created_at, JobTask.job_id).limit(20).all()

        for task in candidates:
            claimable = or_(
                JobTask.status == "QUEUED",
                and_(JobTask.status == "RUNNING", JobTask.lease_expires_at <= now),
            )
            result = db.execute(
                update(JobTask)
                .where(JobTask.job_id == task.job_id, claimable, JobTask.job_id.in_(active_job_ids))
                .values(
                    status="RUNNING",
                    worker_id=worker_id,
                    lease_expires_at=expires,
                    attempts=JobTask.attempts + 1,
                    updated_at=now,
                )
            )
            if result.rowcount == 1:
                db.commit()
                return task.job_id
            db.rollback()
        return None


def renew_lease(job_id: str, worker_id: str, lease_seconds: int = 90) -> bool:
    now = datetime.utcnow()
    with SessionLocal() as db:
        result = db.execute(
            update(JobTask)
            .where(JobTask.job_id == job_id, JobTask.status == "RUNNING", JobTask.worker_id == worker_id)
            .values(lease_expires_at=now + timedelta(seconds=lease_seconds), updated_at=now)
        )
        db.commit()
        return result.rowcount == 1


def finish_task(job_id: str, worker_id: str) -> None:
    with SessionLocal() as db:
        db.execute(
            update(JobTask)
            .where(JobTask.job_id == job_id, JobTask.worker_id == worker_id)
            .values(status="DONE", lease_expires_at=None, updated_at=datetime.utcnow())
        )
        db.commit()


def run_worker(poll_seconds: float = 2.0, once: bool = False, lease_seconds: int = 90) -> None:
    worker_id = f"{socket.gethostname()}-{uuid.uuid4()}"
    adopted = recover_jobs_without_tasks()
    if adopted:
        logger.info("Recovered %d legacy jobs without queue records", adopted)

    logger.info("Durable job worker started: %s", worker_id)
    while True:
        job_id = claim_next_job(worker_id, lease_seconds)
        if job_id is None:
            if once:
                return
            time.sleep(poll_seconds)
            continue

        logger.info("Claimed job %s", job_id)
        try:
            # The runner invokes LangGraph synchronously. Renew the lease while
            # it runs so a second worker cannot reclaim a healthy long job.
            from app.routers.jobs import process_job

            import threading
            stop_heartbeat = threading.Event()

            def heartbeat():
                interval = max(1, lease_seconds // 3)
                while not stop_heartbeat.wait(interval):
                    if not renew_lease(job_id, worker_id, lease_seconds):
                        logger.error("Worker lost lease for job %s", job_id)
                        return

            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                process_job(job_id)
            finally:
                stop_heartbeat.set()
                heartbeat_thread.join(timeout=2)
            finish_task(job_id, worker_id)
        except Exception:
            logger.exception("Worker failed while handling job %s", job_id)
            # Leave the lease to expire so another worker can retry after a
            # process crash or transient database/service failure.
        if once:
            return

