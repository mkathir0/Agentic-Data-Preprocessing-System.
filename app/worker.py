"""Run the durable dataset-job worker as a separate process."""
import argparse
import logging

from app.database import Base, engine
from app.models import JobTask  # noqa: F401 - register queue table in metadata
from app.job_queue import run_worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Process durable AI data engineering jobs")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--lease-seconds", type=int, default=90)
    parser.add_argument("--once", action="store_true", help="Process at most one available job and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    Base.metadata.create_all(bind=engine)
    run_worker(poll_seconds=args.poll_seconds, once=args.once, lease_seconds=args.lease_seconds)


if __name__ == "__main__":
    main()
