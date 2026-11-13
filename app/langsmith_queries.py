"""SmithDB-backed reads for the job trace dashboard."""
import json
from contextlib import closing
from datetime import datetime, timedelta, timezone

from langsmith import Client


async def fetch_job_runs(job_id: str, created_at: datetime | None, endpoint: str, api_key: str, project_name: str):
    """Find a job's root and return its complete trace, including historical jobs."""
    # SQLite stores naive UTC timestamps. Avoid the query API's implicit
    # 24-hour window while allowing for minor timestamp skew at job creation.
    earliest = created_at or datetime(1970, 1, 1, tzinfo=timezone.utc)
    if earliest.tzinfo is None:
        earliest = earliest.replace(tzinfo=timezone.utc)
    earliest -= timedelta(minutes=5)
    with closing(Client(api_url=endpoint, api_key=api_key, timeout_ms=10000)) as client:
        project = await client.aread_project(project_name=project_name)
        root = None
        async for run in client.runs.query(
            project_ids=[str(project.id)],
            is_root=True,
            filter=f'and(eq(input_key, "job_id"), eq(input_value, {json.dumps(job_id)}))',
            min_start_time=earliest,
            selects=["ID", "TRACE_ID"],
            page_size=1,
        ):
            root = run
            break
        if root is None:
            return []
        # This dedicated API returns all runs in a single items response;
        # unlike runs.query, it has no cursor or implicit 24-hour window.
        response = await client.traces.list_runs(
            str(root.trace_id),
            project_id=str(project.id),
            selects=[
                "ID", "TRACE_ID", "PARENT_RUN_IDS", "NAME", "START_TIME",
                "END_TIME", "ERROR", "TAGS", "INPUTS", "OUTPUTS",
            ],
        )
        return response.items
