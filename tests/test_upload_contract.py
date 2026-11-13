"""End-to-end contract tests for uploaded CSVs and production approval."""
import asyncio
from contextlib import ExitStack
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
import io as text_io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi import UploadFile
from langsmith import tracing_context
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import job_queue
from app.agents.cleaning_policy import apply_cleaning_plan, frames_match
from app.agents.schemas import TransformationAction, TransformationPlan
from app.database import Base
from app.models import Job, JobTask
from app.routers import jobs
from app.agents.graph import run_pipeline
from app.config import Settings
from app.utils.data_loader import load_dataframe


class UploadedCsvContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="upload-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.addCleanup(self.engine.dispose)
        self.settings = SimpleNamespace(
            UPLOAD_DIR=str(self.root / "uploads"),
            OUTPUT_DIR=str(self.root / "outputs"),
        )
        self.trace_context = tracing_context(enabled=False)
        self.trace_context.__enter__()
        self.addCleanup(self.trace_context.__exit__, None, None, None)
        env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})
        env.start()
        self.addCleanup(env.stop)
        client = patch("langsmith.run_trees.get_cached_client", return_value=MagicMock())
        client.start()
        self.addCleanup(client.stop)

    def upload_and_process(self, filename: str, raw: bytes, plan: TransformationPlan | None = None):
        with self.factory() as db, patch.object(jobs, "settings", self.settings):
            uploaded = asyncio.run(jobs.upload_files(
                [UploadFile(filename=filename, file=io.BytesIO(raw))], None, db
            ))[0]
            job_id = uploaded.id
            original_path = Path(uploaded.original_filepath)
            original_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()

        patches = [
            patch.object(jobs, "settings", self.settings),
            patch.object(jobs, "SessionLocal", self.factory),
            patch.object(job_queue, "SessionLocal", self.factory),
            patch("app.agents.planner.get_settings", return_value=SimpleNamespace(GROQ_API_KEY="")),
            patch("app.agents.code_generator.get_settings", return_value=SimpleNamespace(GROQ_API_KEY="")),
        ]
        if plan is not None:
            patches.append(patch("app.agents.graph.generate_transformation_plan", return_value=plan))

        with ExitStack() as stack:
            for context in patches:
                stack.enter_context(context)
            claimed = job_queue.claim_next_job("integration-test-worker", lease_seconds=60)
            self.assertEqual(claimed, job_id)
            jobs.process_job(claimed)
            job_queue.finish_task(claimed, "integration-test-worker")

        with self.factory() as db:
            job = db.query(Job).filter_by(id=job_id).one()
            task = db.query(JobTask).filter_by(job_id=job_id).one()
            snapshot = {
                "id": job.id,
                "status": job.status,
                "step": job.current_step,
                "error": job.error_message,
                "raw": job.original_filepath,
                "cleaned": job.cleaned_filepath,
                "report": job.profile_report_path,
                "validation_path": job.validation_report_path,
                "suite_path": job.quality_report_path,
                "validation": job.validation_report,
                "task_status": task.status,
            }
        self.assertEqual(hashlib.sha256(Path(snapshot["raw"]).read_bytes()).hexdigest(), original_hash)
        return snapshot

    def test_uploaded_csv_is_transformed_validated_and_downloadable_only_when_approved(self):
        raw = b"record_id,name,value\n001, Alice ,10\n002,Bob,20\n002,Bob,20\n"
        job = self.upload_and_process("records.csv", raw)

        self.assertEqual(job["status"], "COMPLETED")
        self.assertEqual(job["step"], "COMPLETED")
        self.assertEqual(job["task_status"], "DONE")
        self.assertTrue(job["validation"]["integrity_passed"])
        self.assertTrue(job["validation"]["overall_passed"])
        self.assertFalse(job["validation"]["review_required"])
        self.assertTrue(Path(job["report"]).is_file())
        self.assertTrue(Path(job["validation_path"]).is_file())

        cleaned = load_dataframe(job["cleaned"])
        self.assertEqual(cleaned.to_dict(orient="records"), [
            {"record_id": "001", "name": "Alice", "value": 10},
            {"record_id": "002", "name": "Bob", "value": 20},
        ])
        original = load_dataframe(job["raw"])
        expected, review = apply_cleaning_plan(original, [
            {"action_type": "strip_whitespace", "column": None, "confidence": "high"},
            {"action_type": "drop_duplicates", "column": None, "confidence": "high"},
        ])
        self.assertFalse(review)
        self.assertTrue(frames_match(expected, cleaned))

        with self.factory() as db:
            downloaded = jobs.download_cleaned_csv(job["id"], db)
        self.assertTrue(Path(downloaded.path).is_file())
        self.assertTrue(downloaded.filename.startswith("cleaned_"))

    def test_uploaded_ambiguous_csv_is_preserved_and_never_production_approved(self):
        raw = b"record_id,amount\n001,-50\n002,100\n"
        plan = TransformationPlan(
            dataset_summary="The meaning of negative amounts is unknown.",
            actions=[TransformationAction(
                column="amount",
                action_type="flag_for_review",
                explanation="Negative amounts could be credits or errors.",
                confidence="low",
            )],
        )
        job = self.upload_and_process("ambiguous.csv", raw, plan)

        self.assertEqual(job["status"], "FAILED")
        self.assertEqual(job["step"], "REVIEW_FLAGGED")
        self.assertTrue(job["validation"]["integrity_passed"])
        self.assertTrue(job["validation"]["review_required"])
        self.assertFalse(job["validation"]["overall_passed"])
        self.assertIsNotNone(job["cleaned"])
        pd.testing.assert_frame_equal(load_dataframe(job["raw"]), load_dataframe(job["cleaned"]))
        with self.factory() as db:
            downloaded = jobs.download_cleaned_csv(job["id"], db)
        self.assertTrue(downloaded.filename.startswith("review_candidate_"))

    def test_uploaded_format_mess_is_normalized_and_receives_approval_when_validation_passes(self):
        raw = (
            b"record_id,unit_price,order_date,is_active\n"
            b"001,$10.00,02/13/2024,yes\n"
            b"002,USD 20.00,2024-03-14,false\n"
        )
        job = self.upload_and_process("format-mess.csv", raw)
        self.assertEqual(job["status"], "COMPLETED")
        self.assertEqual(job["step"], "COMPLETED")
        self.assertTrue(job["validation"]["integrity_passed"])
        self.assertTrue(job["validation"]["overall_passed"])
        cleaned = load_dataframe(job["cleaned"])
        self.assertEqual(cleaned["unit_price"].tolist(), [10.0, 20.0])
        self.assertEqual(cleaned["order_date"].tolist(), [
            "2024-02-13T00:00:00", "2024-03-14T00:00:00",
        ])
        self.assertEqual(cleaned["is_active"].tolist(), ["true", "false"])
        with self.factory() as db:
            downloaded = jobs.download_cleaned_csv(job["id"], db)
        self.assertTrue(downloaded.filename.startswith("cleaned_"))

    def test_unparseable_uploaded_csv_fails_closed_without_cleaned_output(self):
        raw = b"record_id,record_id\n001,002\n"
        job = self.upload_and_process("duplicate-headers.csv", raw)
        self.assertEqual(job["status"], "FAILED")
        self.assertIsNone(job["cleaned"])
        self.assertIn("Duplicate column names", job["error"])

    def test_supported_policy_round_trips_generated_messy_csv_matrix(self):
        actions = [
            {"action_type": "strip_whitespace", "column": None, "confidence": "high"},
            {"action_type": "drop_duplicates", "column": None, "confidence": "high"},
        ]
        for case in range(30):
            with self.subTest(case=case):
                before = pd.DataFrame({
                    "record_id": [f"{case:02d}-{i:03d}" for i in range(12)] + [f"{case:02d}-000"],
                    "label": [f" value-{(i * (case + 1)) % 5} " if i % 3 else "NA" for i in range(12)] + [" value-0 "],
                    "amount": [(i * 13 + case) for i in range(12)] + [case],
                })
                expected, review = apply_cleaning_plan(before, actions)
                csv_path = self.root / f"case-{case}.csv"
                output_path = self.root / f"case-{case}-out.csv"
                before.to_csv(csv_path, index=False)
                expected.to_csv(output_path, index=False)
                actual = load_dataframe(str(output_path))
                self.assertFalse(review)
                self.assertTrue(frames_match(expected, actual))
                self.assertEqual(list(actual.columns), list(before.columns))
                self.assertEqual(len(actual), len(expected))

    def test_benchmark_uses_same_policy_and_never_persists_cleaned_candidates(self):
        from benchmark import run_benchmarks

        raw_path = self.root / "fixture.csv"
        pd.DataFrame({"record_id": ["001", "002", "002"], "label": [" A ", "B", "B"]}).to_csv(raw_path, index=False)
        summary_path = self.root / "summary.json"
        report_path = self.root / "benchmark.md"
        summary = run_benchmarks(
            benchmark_dir=self.root,
            summary_path=summary_path,
            report_path=report_path,
            benchmark_files=["fixture.csv"],
        )
        result = summary["datasets"][0]
        self.assertTrue(result["integrity_passed"])
        self.assertTrue(summary_path.is_file())
        self.assertTrue(report_path.is_file())
        self.assertFalse((self.root / "cleaned_fixture.csv").exists())
        self.assertFalse(summary["production_approval"])

    def test_generated_massive_fixture_runs_expected_safe_cleanup_and_flags_injected_defects(self):
        repo_root = Path(__file__).resolve().parents[1]
        contract_path = repo_root / "tests" / "fixtures" / "massive_messy_dataset_expectations.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        source = repo_root / contract["source"]
        if not source.is_file():
            self.skipTest("repository massive messy fixture is unavailable")
        raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        before = load_dataframe(str(source))
        expected_exact_duplicates = int(before.duplicated().sum())
        output = self.root / "massive_cleaned.csv"
        report_path = self.root / "massive_report.md"
        settings = Settings(GROQ_API_KEY="")
        with patch("app.agents.planner.get_settings", return_value=settings), patch(
            "app.agents.code_generator.get_settings", return_value=settings
        ):
            result = run_pipeline(
                "massive-fixture-regression", str(source), str(output), str(report_path)
            )

        after = load_dataframe(str(output))
        validation = result["validation_report"]
        plan = json.loads(result["plan_json"])
        actions = plan["actions"]
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), raw_hash)
        self.assertEqual(len(before), contract["expected_input_rows"])
        self.assertEqual(len(before.columns), contract["expected_columns"])
        finding_ids = {finding["id"] for finding in contract["required_validation_findings"]}
        for category in contract["generator_anomaly_coverage"]:
            with self.subTest(generator_category=category["category"]):
                self.assertTrue(set(category.get("findings", [])).issubset(finding_ids))
                action_types = {action["action_type"] for action in actions}
                self.assertTrue(set(category.get("transformations", [])).issubset(action_types))
                if "transformation" in category:
                    self.assertIn(category["transformation"], contract["expected_transformation"])
        self.assertEqual(len(before) - len(after), expected_exact_duplicates)
        expected_removed = contract["permitted_transformations"]["expected_exact_duplicate_rows_removed"]
        self.assertEqual(expected_exact_duplicates, expected_removed)
        self.assertEqual(validation["exact_duplicates_removed"], expected_removed)
        for column in before.select_dtypes(include=["object", "string"]).columns:
            values = after[column].dropna().astype(str)
            self.assertFalse((values != values.str.strip()).any(), column)
        expected, policy_review = apply_cleaning_plan(before, actions)
        self.assertFalse(policy_review)
        self.assertTrue(frames_match(expected, after))
        expected_columns = contract["expected_transformation"]
        action_columns = {(action["action_type"], action["column"]) for action in actions}
        for col in expected_columns["normalized_numeric_columns"]:
            self.assertIn(("normalize_numeric_format", col), action_columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(after[col]), col)
        for col in expected_columns["normalized_datetime_columns"]:
            self.assertIn(("normalize_datetime_format", col), action_columns)
            dates = after[col].dropna().astype(str)
            self.assertTrue(dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}").all(), col)
        for col in expected_columns["normalized_boolean_columns"]:
            self.assertIn(("normalize_boolean", col), action_columns)
            self.assertLessEqual(set(after[col].dropna().astype(str)), {"true", "false"})
        for col in expected_columns["case_only_category_columns"]:
            self.assertIn(("normalize_category_case", col), action_columns)
            values = after[col].dropna().astype(str)
            self.assertEqual(len(values.str.casefold().unique()), values.nunique())
        for key, expected in contract["approval_expectation"].items():
            actual = {
                "integrity_passed": validation["integrity_passed"],
                "overall_passed": validation["overall_passed"],
                "review_required": validation["review_required"],
                "production_approved": validation["overall_passed"] and validation["integrity_passed"],
            }[key]
            self.assertEqual(actual, expected, key)
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["final_status"], "FAILED")
        for finding in contract["required_validation_findings"]:
            with self.subTest(finding=finding["id"]):
                if finding["source"] == "checks":
                    matches = [
                        check
                        for field in finding["fields"]
                        for check in validation[field]
                        if finding["signal"] in check["name"] or finding["signal"] in check["details"]
                    ]
                    self.assertTrue(matches, f"Missing expected finding: {finding['id']}")
                    self.assertGreaterEqual(
                        sum(check["violations"] for check in matches),
                        finding["minimum_violations"],
                        f"Finding below expected minimum: {finding['id']}",
                    )
                elif finding["source"] == "stats_diff":
                    diff = next((d for d in validation["stats_diff"] if d["column"] == finding["column"]), None)
                    self.assertIsNotNone(diff, f"Missing profile statistics for {finding['id']}")
                    self.assertGreaterEqual(diff["before"]["null_pct"], finding["minimum_null_fraction"])
        self.assertTrue(report_path.is_file())

    def test_generator_output_runs_through_pipeline_without_guessing_at_defects(self):
        """Exercise generated messy input end to end, including safe cleanup and review gates."""
        from generate_messy_dataset import create_massive_messy_dataset

        source = self.root / "generated-messy.csv"
        with redirect_stdout(text_io.StringIO()):
            generated = create_massive_messy_dataset(num_rows=400, output_path=str(source))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        before = load_dataframe(str(source))
        self.assertEqual(generated.shape, (870, 22))  # 400 base + 350 exact dupes + 120 conflicts
        self.assertEqual(int(before.duplicated().sum()), 350)

        output = self.root / "generated-messy-cleaned.csv"
        report_path = self.root / "generated-messy-report.md"
        settings = Settings(GROQ_API_KEY="")
        with patch("app.agents.planner.get_settings", return_value=settings), patch(
            "app.agents.code_generator.get_settings", return_value=settings
        ):
            result = run_pipeline(
                "generated-messy-regression", str(source), str(output), str(report_path)
            )

        after = load_dataframe(str(output))
        validation = result["validation_report"]
        actions = json.loads(result["plan_json"])["actions"]
        expected, review = apply_cleaning_plan(before, actions)
        self.assertFalse(review)
        self.assertTrue(frames_match(expected, after))
        self.assertEqual(len(before) - len(after), 350)
        self.assertEqual(validation["exact_duplicates_removed"], 350)
        self.assertTrue(validation["integrity_passed"])
        self.assertTrue(validation["review_required"])
        self.assertFalse(validation["overall_passed"])
        self.assertEqual(result["final_status"], "FAILED")
        # Reused transaction IDs, malformed emails, and other semantic defects
        # remain data-quality findings; the cleaner must not invent corrections.
        self.assertGreater(after["transaction_id"].duplicated().sum(), 0)
        malformed_email = before["customer_email"].dropna().astype(str).loc[
            lambda values: ~values.str.contains("@")
        ]
        self.assertGreater(len(malformed_email), 0)
        self.assertTrue(set(malformed_email).issubset(set(after["customer_email"].dropna().astype(str))))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
        self.assertTrue(report_path.is_file())

    def test_messy_generator_is_repeatable_and_injects_declared_anomalies(self):
        from generate_messy_dataset import create_massive_messy_dataset

        first_path = self.root / "generator-first.csv"
        second_path = self.root / "generator-second.csv"
        with redirect_stdout(text_io.StringIO()):
            first = create_massive_messy_dataset(num_rows=400, output_path=str(first_path))
            second = create_massive_messy_dataset(num_rows=400, output_path=str(second_path))

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(hashlib.sha256(first_path.read_bytes()).hexdigest(), hashlib.sha256(second_path.read_bytes()).hexdigest())
        self.assertEqual(len(first), 870)
        self.assertEqual(int(first.duplicated().sum()), 350)
        conflicts = first.groupby("transaction_id")["customer_name"].nunique()
        self.assertGreater(int((conflicts > 1).sum()), 0)
        emails = first["customer_email"].dropna().astype(str)
        self.assertTrue(emails.str.contains("gmail.com", regex=False).any())
        self.assertTrue((emails.str.contains("gmail.com", regex=False) & ~emails.str.contains("@", regex=False)).any())
        self.assertTrue(first["unit_price"].astype(str).str.contains(r"\$|USD", regex=True).any())
        self.assertGreater((pd.to_numeric(first["customer_rating"], errors="coerce") > 5).sum(), 0)
        self.assertTrue(first["order_date"].astype(str).str.contains("/").any())


if __name__ == "__main__":
    unittest.main()
