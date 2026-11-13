import asyncio
import os
import io
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi import HTTPException, UploadFile
from langsmith import tracing_context
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.cleaning_policy import apply_cleaning_plan, frames_match, suggest_safe_format_actions
from app.agents.code_generator import _generate_heuristic_code
from app.agents.executor import execute_cleaning_code, _sandbox_command
from app.agents.schema_agent import run_schema_inference
from app.agents.schemas import TransformationPlan, TransformationAction
from app.agents.validation_engine import execute_full_validation_engine, build_dynamic_pandera_schema, export_great_expectations_suite
from app.agents.graph import route_after_validation, node_validate
from app.agents.reporter import generate_report
from app.database import Base
from app.models import Job, JobTask
from app import job_queue
from app.routers import jobs
from app.utils.data_loader import load_dataframe


def action(kind, column=None, confidence='high'):
    return dict(action_type=kind, column=column, confidence=confidence, explanation='Test rule')


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, {'LANGSMITH_TRACING': 'false', 'LANGCHAIN_TRACING_V2': 'false'})
        env.start(); self.addCleanup(env.stop)
        client = patch('langsmith.run_trees.get_cached_client', return_value=MagicMock())
        client.start(); self.addCleanup(client.stop)
        self.trace = tracing_context(enabled=False)
        self.trace.__enter__()
        self.addCleanup(self.trace.__exit__, None, None, None)

    def save(self, name, frame):
        path = self.root / name
        frame.to_csv(path, index=False)
        return str(path)

    def validate(self, before, after, actions=None):
        return execute_full_validation_engine(self.save('before.csv', before), self.save('after.csv', after), plan_actions=actions or [])

    def test_loader_preserves_identifiers_and_literal_null_tokens(self):
        p = self.root / 'input.csv'
        p.write_text('customer_id,country,token,value\n00123,NA,NULL,10\n00456,US,nan,20\n')
        df = load_dataframe(str(p))
        self.assertEqual(df.customer_id.tolist(), ['00123', '00456'])
        self.assertEqual(df.country.tolist(), ['NA', 'US'])
        self.assertEqual(df.token.tolist(), ['NULL', 'nan'])
        self.assertEqual(df.value.tolist(), [10, 20])

    def test_loader_preserves_noncanonical_numbers_and_precision(self):
        p = self.root / 'input.csv'
        p.write_text('zip,precise,amount\n00123,9007199254740993,10.00\n00456,9007199254740995,20.00\n')
        df = load_dataframe(str(p))
        self.assertEqual(df.iloc[0].tolist(), ['00123', '9007199254740993', '10.00'])

    def test_loader_rejects_duplicate_headers(self):
        p = self.root / 'input.csv'; p.write_text('a,a\n1,2\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_dataframe(str(p))

    def test_schema_does_not_treat_unique_measures_as_primary_keys(self):
        path = self.save('schema.csv', pd.DataFrame({'salary': [45000, 55000, 65000], 'age': [20, 30, 40]}))
        self.assertEqual(run_schema_inference(path).primary_keys, [])

    def test_schema_identifies_named_key(self):
        path = self.save('schema.csv', pd.DataFrame({'customer_id': ['001', '002'], 'salary': [40000, 50000]}))
        self.assertEqual(run_schema_inference(path).primary_keys, ['customer_id'])

    def test_fallback_respects_review_and_does_not_invent_values(self):
        before = pd.DataFrame({'record_id': [1, 2, 3], 'amount': [-50., 100., None], 'age': [130, 25, 30]})
        plan = TransformationPlan(dataset_summary='Preserve', actions=[TransformationAction(**action('flag_for_review', 'amount', 'low'))])
        namespace = {}; exec(_generate_heuristic_code(plan).code, namespace)
        pd.testing.assert_frame_equal(namespace['clean_data'](before), before)

    def test_supported_plan_formats_and_deduplicates(self):
        before = pd.DataFrame({'record_id': ['001', '001'], 'name': [' Alice ', 'Alice']})
        after, review = apply_cleaning_plan(before, [action('strip_whitespace', 'name'), action('drop_duplicates')])
        self.assertEqual(len(after), 1)
        self.assertEqual(after.iloc[0].tolist(), ['001', 'Alice'])
        self.assertEqual(review, [])

    def test_common_representation_repairs_are_deterministic_and_validatable(self):
        before = pd.DataFrame({
            'record_id': [f'{i:03}' for i in range(80)],
            'unit_price': ['$1,200.00', 'USD 30.00', 'N/A', '-$5.00'] * 20,
            'order_date': ['2024-01-02', '13/01/2024', '14-Jan-2024', '1705320000'] * 20,
            'is_active': ['Y', 'no', 'TRUE', '0'] * 20,
            'category': ['Books', 'books', ' BOOKS ', 'Music'] * 20,
        })
        actions = [
            action('strip_whitespace'),
            {**action('normalize_numeric_format', 'unit_price'), 'parameters': {}},
            {**action('normalize_datetime_format', 'order_date'), 'parameters': {'date_order': 'day_first'}},
            action('normalize_boolean', 'is_active'),
            action('normalize_category_case', 'category'),
        ]
        after, review = apply_cleaning_plan(before, actions)
        self.assertFalse(review)
        self.assertEqual(after.unit_price.iloc[0], 1200.0)
        self.assertEqual(after.unit_price.iloc[1], 30.0)
        self.assertTrue(pd.isna(after.unit_price.iloc[2]))
        self.assertEqual(after.unit_price.iloc[3], -5.0)
        self.assertEqual(after.order_date.iloc[1], '2024-01-13T00:00:00')
        self.assertEqual(set(after.is_active), {'true', 'false'})
        self.assertEqual(after.category.iloc[:3].tolist(), ['Books', 'Books', 'Books'])
        report = self.validate(before, after, actions)
        self.assertTrue(report.integrity_passed)
        self.assertEqual(report.new_nulls_introduced, {})

    def test_ambiguous_dates_are_preserved_and_escalated(self):
        before = pd.DataFrame({'date': ['01/02/2024', '03/04/2024']})
        after, review = apply_cleaning_plan(before, [
            {**action('normalize_datetime_format', 'date'), 'parameters': {'date_order': None}}
        ])
        pd.testing.assert_frame_equal(after, before)
        self.assertTrue(review)

    def test_repairs_without_deterministic_evidence_are_rejected(self):
        before = pd.DataFrame({
            'customer_email': ['jane@gmial.com', 'JANE@example.com'],
            'transaction_id': ['TX-1', 'TX-1'],
            'amount': ['$10.00', '$20.00'],
            'date': ['01/02/2024', '03/04/2024'],
        })
        actions = [
            action('normalize_category_case', 'customer_email'),
            action('repair_email', 'customer_email'),
            action('reconcile_transaction_id', 'transaction_id'),
            {**action('normalize_datetime_format', 'date'), 'parameters': {'date_order': 'month_first'}},
        ]
        after, review = apply_cleaning_plan(before, actions)
        pd.testing.assert_frame_equal(after, before)
        self.assertEqual(len(review), len(actions))
        self.assertTrue(all('preserved' in reason or 'reviewed domain rule' in reason for reason in review))

    def test_format_repair_action_must_match_profiler_parameters(self):
        before = pd.DataFrame({'order_date': ['13/01/2024', '2024-01-14']})
        after, review = apply_cleaning_plan(before, [
            {**action('normalize_datetime_format', 'order_date'), 'parameters': {'date_order': 'month_first'}}
        ])
        pd.testing.assert_frame_equal(after, before)
        self.assertTrue(review)

    def test_profiler_candidates_skip_ambiguous_dates_and_identifiers(self):
        before = pd.DataFrame({
            'record_id': ['001', '002'],
            'date': ['01/02/2024', '03/04/2024'],
            'unit_price': ['$10.00', '$20.00'],
            'is_live': ['yes', '1'],
            'category': ['Books', 'books'],
        })
        actions = suggest_safe_format_actions(before, ['category'])
        pairs = {(a['action_type'], a['column']) for a in actions}
        self.assertIn(('normalize_numeric_format', 'unit_price'), pairs)
        self.assertIn(('normalize_boolean', 'is_live'), pairs)
        self.assertIn(('normalize_category_case', 'category'), pairs)
        self.assertNotIn(('normalize_datetime_format', 'date'), pairs)
        self.assertFalse(any(a['column'] == 'record_id' for a in actions))

    def test_numeric_sentinel_policy_does_not_destroy_a_valid_negative_number(self):
        before = pd.DataFrame({'amount': ['-999', '$10.00']})
        after, review = apply_cleaning_plan(before, [
            {**action('normalize_numeric_format', 'amount'), 'parameters': {}}
        ])
        self.assertFalse(review)
        self.assertEqual(after.amount.tolist(), [-999.0, 10.0])

    def test_percent_scale_is_not_inferred_without_matching_evidence(self):
        ambiguous = pd.DataFrame({'discount_pct': ['0.15', '10%']})
        self.assertFalse(any(
            action['action_type'] == 'normalize_numeric_format'
            for action in suggest_safe_format_actions(ambiguous)
        ))

    def test_review_column_cannot_be_changed_by_global_strip(self):
        before = pd.DataFrame({'name': [' Alice '], 'city': [' Paris ']})
        after, review = apply_cleaning_plan(before, [action('flag_for_review', 'name'), action('strip_whitespace')])
        self.assertEqual(after.name.iloc[0], ' Alice ')
        self.assertEqual(after.city.iloc[0], 'Paris')
        self.assertTrue(review)

    def test_plan_does_not_deduplicate_by_subset_or_impute(self):
        before = pd.DataFrame({'record_id': ['1', '1'], 'value': [10, None]})
        after, review = apply_cleaning_plan(before, [action('drop_duplicates', 'record_id'), action('fill_missing_median', 'value')])
        pd.testing.assert_frame_equal(after, before)
        self.assertEqual(len(review), 2)

    def test_validation_rejects_unplanned_data_manipulation(self):
        before = pd.DataFrame({'record_id': range(1, 101), 'value': [10]*100})
        candidates = {
            'deleted_rows': before.iloc[:90],
            'dropped_column': before.drop(columns='value'),
            'invented_rows': pd.concat([before, pd.DataFrame({'record_id': range(101, 111), 'value': [10]*10})]),
            'changed_value': before.assign(value=[11]+[10]*99),
            'changed_key': before.assign(record_id=[101]+list(range(2,101))),
        }
        for name, after in candidates.items():
            with self.subTest(name=name):
                report = self.validate(before, after)
                self.assertFalse(report.overall_passed)
                self.assertFalse(report.integrity_passed)
                self.assertEqual(report.exact_duplicates_removed, 0)

    def test_validation_allows_reordering_without_changing_multiplicities(self):
        before = pd.DataFrame({'value': [1, 2, 2]})
        self.assertTrue(frames_match(before, before.iloc[::-1]))
        self.assertFalse(frames_match(before, pd.DataFrame({'value': [1, 1, 2]})))

    def test_planned_exact_duplicate_removal_is_counted_truthfully(self):
        before = pd.DataFrame({'value': [10, 10, 20]})
        report = self.validate(before, before.drop_duplicates(), [action('drop_duplicates')])
        self.assertTrue(report.integrity_passed)
        self.assertEqual(report.exact_duplicates_removed, 1)
        self.assertFalse(self.validate(before, before.drop_duplicates()).overall_passed)

    def test_unsupported_action_requires_review_even_if_values_unchanged(self):
        before = pd.DataFrame({'value': [10, 20]})
        report = self.validate(before, before, [action('fill_missing_median', 'value')])
        self.assertTrue(report.integrity_passed)
        self.assertTrue(report.review_required)
        self.assertFalse(report.overall_passed)

    def test_valid_unchanged_data_passes_with_input_contract(self):
        before = pd.DataFrame({'record_id': ['001', '002'], 'value': [10, 20]})
        report = self.validate(before, before)
        self.assertTrue(report.integrity_passed)
        self.assertTrue(report.overall_passed)

    def test_validation_retry_routes_back_through_planner(self):
        self.assertEqual(route_after_validation({'error_feedback': 'fix evidence', 'retry_count': 1}), 'plan')
        self.assertEqual(route_after_validation({'error_feedback': 'still failing', 'retry_count': 3}), 'report')

    def test_integrity_verified_review_finding_does_not_trigger_retries(self):
        report = SimpleNamespace(
            review_required=True,
            total_failed=2,
            pandera_schema_passed=False,
            integrity_passed=True,
            overall_passed=False,
            escalation_flags=[],
            pandera_failure_cases=[],
            new_nulls_introduced={},
            key_conflict_count=0,
            key_conflict_details=[],
            regression_checks=[],
            schema_checks=[], range_checks=[], cross_field_checks=[],
            distributional_checks=[], duplicate_checks=[], format_checks=[],
            review_reasons=['Missing values preserved for review'],
            total_passed=4, total_checks=4, data_quality_score_after=80.0,
            score_improvement=0.0,
            model_dump=lambda: {
                'review_required': True, 'overall_passed': False,
                'integrity_passed': True, 'review_reasons': ['Missing values preserved for review'],
            },
        )
        state = {
            'input_file': 'raw.csv', 'output_file': 'candidate.csv',
            'schema_data': None, 'plan_json': '{}', 'profiler_data': {},
        }
        with patch('app.agents.graph.run_full_validation', return_value=report):
            result = node_validate(state)
        self.assertIsNone(result['error_feedback'])
        self.assertFalse(result['validation_report']['overall_passed'])
        self.assertTrue(result['validation_report']['integrity_passed'])

    def test_heuristic_planner_keeps_review_findings_out_of_safe_transform_actions(self):
        from app.agents.planner import _generate_heuristic_plan
        from app.agents.schemas import ProfilerOutput, QualityOutput, QualityAnomaly, SchemaOutput
        profiler = ProfilerOutput(
            num_rows=3, num_columns=1, memory_usage_bytes=0, missing_cells=1,
            missing_cells_percentage=33.3, duplicate_rows=0,
            duplicate_rows_percentage=0.0, column_types={'amount': 'Numeric'},
            columns_summary={},
        )
        quality = QualityOutput(
            is_valid=True,
            anomalies=[
                QualityAnomaly(column='amount', anomaly_type='missing_values', description='One missing value', severity='low'),
                QualityAnomaly(column=None, anomaly_type='cross_field_violation', description='Date ordering conflict', severity='high'),
            ],
            summary='Missing values and a cross-field finding',
        )
        schema = SchemaOutput(primary_keys=[], candidate_foreign_keys=[], categorical_columns=[], datetime_columns=[], inferred_types={})
        plan = _generate_heuristic_plan(profiler, quality, schema)
        self.assertEqual([action.action_type for action in plan.actions], ['strip_whitespace', 'drop_duplicates'])
        before = pd.DataFrame({'name': [' A ', ' A '], 'amount': [10, 10]})
        cleaned, review = apply_cleaning_plan(before, [action.model_dump() for action in plan.actions])
        self.assertEqual(cleaned.to_dict(orient='records'), [{'name': 'A', 'amount': 10}])
        self.assertFalse(review)

    def test_report_writes_machine_readable_validation_and_suite_artifacts(self):
        output = self.root / 'report_input.md'
        suite = {'expectations': [{'type': 'expect_table_row_count_to_be_between'}]}
        report = {
            'overall_passed': False,
            'integrity_passed': True,
            'review_required': True,
            'review_reasons': ['Needs domain review'],
            'great_expectations_suite': suite,
        }
        generate_report('job', None, None, 'Review needed', str(output), report)
        self.assertTrue((self.root / 'validation_input.json').is_file())
        self.assertTrue((self.root / 'quality_suite_input.json').is_file())
        self.assertIn('NOT APPROVED', output.read_text())
        self.assertIn('review required', output.read_text().lower())

    def test_export_uses_real_regex_and_bounds(self):
        schema = build_dynamic_pandera_schema(pd.DataFrame({'email': ['alice@example.com'], 'age': [30]}))
        suite = export_great_expectations_suite(schema)
        regex = next(e['kwargs']['regex'] for e in suite['expectations'] if e['expectation_type'] == 'expect_column_values_to_match_regex')
        self.assertRegex('alice@example.com', regex)
        self.assertNotRegex('invalid-email', regex)
        bound = next(e['kwargs'] for e in suite['expectations'] if e['expectation_type'] == 'expect_column_values_to_be_between')
        self.assertEqual((bound['min_value'], bound['max_value']), (0., 120.))

    def test_executor_preserves_input_and_quoted_paths(self):
        path = self.root / "O'Brien.csv"; path.write_text('record_id,country\n001,NA\n')
        original = path.read_bytes(); output = self.root / 'output.csv'
        result = execute_cleaning_code('def clean_data(df):\n    return df', str(path), str(output))
        self.assertTrue(result['success'], result['error'])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(load_dataframe(str(output)).iloc[0].tolist(), ['001', 'NA'])

    def test_executor_blocks_host_file_reads_and_writes(self):
        path=self.save('input.csv',pd.DataFrame({'value':[1]}));out=self.root/'out.csv'
        marker=Path('/private/tmp')/f'generated-code-escape-{uuid.uuid4().hex}'
        code=(
            'from pathlib import Path\n'
            'try:\n    Path("/etc/passwd").read_text()\n'
            'except Exception:\n    pass\n'
            'else:\n    raise RuntimeError("host file read was not blocked")\n'
            f'try:\n    Path({str(marker)!r}).write_text("escaped")\n'
            'except Exception:\n    pass\n'
            'else:\n    raise RuntimeError("host file write was not blocked")\n'
            'def clean_data(df):\n    return df\n'
        )
        result=execute_cleaning_code(code,path,str(out))
        self.assertTrue(result['success'],result['error'])
        self.assertFalse(marker.exists())

    def test_failed_execution_removes_stale_output(self):
        path = self.save('input.csv', pd.DataFrame({'value': [1]})); out = self.root/'output.csv';out.write_text('stale\n999\n')
        result = execute_cleaning_code('def clean_data(df):\n    raise ValueError("deliberate")',path,str(out))
        self.assertFalse(result['success'])
        self.assertFalse(out.exists())

    def test_executor_does_not_accept_missing_or_invalid_output(self):
        path = self.save('input.csv', pd.DataFrame({'value':[1]})); out=self.root/'out.csv'
        for code in ['def clean_data(df):\n    return None', 'def clean_data(df):\n    import sys\n    sys.exit(0)']:
            result=execute_cleaning_code(code,path,str(out));self.assertFalse(result['success']);self.assertFalse(out.exists())

    def test_failed_pipeline_cannot_be_completed_by_existing_output(self):
        job = SimpleNamespace(id='job-a', filename='data.csv', original_filepath='input', user_instructions=None)
        db=MagicMock();db.query.return_value.filter.return_value.first.return_value=job
        directory=self.root/'job-a';directory.mkdir();(directory/'cleaned_data.csv').write_text('value\n999\n')
        with patch.object(jobs,'settings',SimpleNamespace(OUTPUT_DIR=str(self.root))), patch.object(jobs,'run_pipeline',return_value={'final_status':'FAILED'}):
            jobs.process_job('job-a',db)
        self.assertEqual(job.status,'FAILED')
        self.assertIsNone(job.cleaned_filepath)

    def test_job_status_requires_verified_validation(self):
        for passed in [False,True]:
            job = SimpleNamespace(id='job-a', filename='data.csv', original_filepath='input', user_instructions=None)
            db=MagicMock();db.query.return_value.filter.return_value.first.return_value=job
            def pipeline(**kwargs):
                Path(kwargs['output_file']).write_text('value\n1\n')
                return {'final_status':'COMPLETED' if passed else 'FAILED','validation_report':{'overall_passed':passed,'integrity_passed':True}}
            with patch.object(jobs,'settings',SimpleNamespace(OUTPUT_DIR=str(self.root))),patch.object(jobs,'run_pipeline',side_effect=pipeline):
                jobs.process_job('job-a',db)
            self.assertEqual(job.status,'COMPLETED' if passed else 'FAILED')
            self.assertEqual(job.current_step,'COMPLETED' if passed else 'REVIEW_FLAGGED')

    def test_uploads_with_same_name_are_isolated_and_queued_durably(self):
        engine=create_engine('sqlite://');Base.metadata.create_all(engine)
        with Session(engine) as db:
            settings=SimpleNamespace(UPLOAD_DIR=str(self.root/'uploads'), OUTPUT_DIR=str(self.root/'outputs'))
            files=[UploadFile(filename='data.csv',file=io.BytesIO(data)) for data in [b'a\n1\n',b'a\n2\n']]
            with patch.object(jobs,'settings',settings):
                records=asyncio.run(jobs.upload_files(files,None,db))
            self.assertNotEqual(records[0].original_filepath,records[1].original_filepath)
            self.assertEqual([Path(r.original_filepath).read_bytes() for r in records],[b'a\n1\n',b'a\n2\n'])
            self.assertEqual([db.query(JobTask).filter_by(job_id=r.id).one().status for r in records],['QUEUED','QUEUED'])
        engine.dispose()

    def test_upload_rejects_directory_components(self):
        settings=SimpleNamespace(UPLOAD_DIR=str(self.root/'uploads'),OUTPUT_DIR=str(self.root/'outputs'))
        for name in ['../data.csv',r'..\data.csv']:
            with patch.object(jobs,'settings',settings),self.assertRaises(HTTPException):
                asyncio.run(jobs.upload_files([UploadFile(filename=name,file=io.BytesIO(b'x\n1\n'))],None,MagicMock()))

    def test_durable_queue_claims_once_and_reclaims_expired_lease(self):
        engine=create_engine('sqlite://')
        Base.metadata.create_all(engine)
        factory=sessionmaker(bind=engine)
        with factory() as db:
            job=Job(id='queued-job',filename='in.csv',status='PENDING')
            db.add(job);db.add(JobTask(job_id=job.id,status='QUEUED'));db.commit()
        with patch.object(job_queue,'SessionLocal',factory):
            self.assertEqual(job_queue.claim_next_job('worker-a',lease_seconds=60),'queued-job')
            self.assertIsNone(job_queue.claim_next_job('worker-b',lease_seconds=60))
            with factory() as db:
                task=db.query(JobTask).filter_by(job_id='queued-job').one()
                task.lease_expires_at=datetime.utcnow()-timedelta(seconds=1)
                db.commit()
            self.assertEqual(job_queue.claim_next_job('worker-b',lease_seconds=60),'queued-job')
        with factory() as db:
            task=db.query(JobTask).filter_by(job_id='queued-job').one()
            self.assertEqual(task.worker_id,'worker-b')
            self.assertEqual(task.attempts,2)
        engine.dispose()

    def test_queue_recovers_legacy_processing_jobs(self):
        engine=create_engine('sqlite://')
        Base.metadata.create_all(engine)
        factory=sessionmaker(bind=engine)
        with factory() as db:
            db.add(Job(id='orphan-job',filename='old.csv',status='PROCESSING',progress=50))
            db.commit()
        with patch.object(job_queue,'SessionLocal',factory):
            self.assertEqual(job_queue.recover_jobs_without_tasks(),1)
        with factory() as db:
            job=db.query(Job).filter_by(id='orphan-job').one()
            task=db.query(JobTask).filter_by(job_id='orphan-job').one()
            self.assertEqual(job.status,'PENDING')
            self.assertEqual(job.progress,0)
            self.assertEqual(task.status,'QUEUED')
        engine.dispose()

    def test_linux_sandbox_command_has_container_isolation(self):
        with patch('app.agents.executor.sys.platform','linux'), patch('app.agents.executor.shutil.which',return_value='/usr/bin/docker'):
            command, container_name = _sandbox_command(self.root,self.root/'run.py',self.root/'in.csv',self.root/'out.csv')
        self.assertIn('--network=none',command)
        self.assertIn('--read-only',command)
        self.assertIn('--cap-drop=ALL',command)
        self.assertIn('--security-opt=no-new-privileges',command)
        self.assertIn('--pids-limit=64',command)
        self.assertTrue(container_name.startswith('data-cleaner-'))

    def test_legacy_output_download_is_blocked_until_reverified(self):
        path=self.save('legacy.csv',pd.DataFrame({'x':[1]}))
        job=SimpleNamespace(status='COMPLETED',cleaned_filepath=path,validation_report={},filename='legacy.csv')
        db=MagicMock();db.query.return_value.filter.return_value.first.return_value=job
        with self.assertRaises(HTTPException) as error:
            jobs.download_cleaned_csv('legacy',db)
        self.assertEqual(error.exception.status_code,409)

    def test_whitespace_only_cells_roundtrip_under_explicit_rule(self):
        before=pd.DataFrame({'label':['   ','ok']})
        after, _=apply_cleaning_plan(before,[action('strip_whitespace')])
        report=self.validate(before,after,[action('strip_whitespace')])
        self.assertTrue(report.integrity_passed)

    def test_numeric_inference_preserves_negative_zero(self):
        p=self.root/'input.csv';p.write_text('value\n-0\n2\n')
        self.assertEqual(load_dataframe(str(p)).value.iloc[0],'-0')

    def test_full_graph_supported_cleaning_and_report(self):
        from app.agents import graph
        source=self.save('input.csv',pd.DataFrame({'record_id':['001','002'],'label':[' Alice ','Bob'],'value':[10,20]}))
        with patch('app.agents.planner.get_settings',return_value=SimpleNamespace(GROQ_API_KEY='')),patch('app.agents.code_generator.get_settings',return_value=SimpleNamespace(GROQ_API_KEY='')):
            result=graph.run_pipeline('test-job',source,str(self.root/'out.csv'),str(self.root/'report.md'))
        self.assertEqual(result['final_status'],'COMPLETED',result['quality_summary'])
        self.assertTrue(result['validation_report']['integrity_passed'])
        self.assertTrue(result['validation_report']['overall_passed'])
        self.assertEqual(load_dataframe(str(self.root/'out.csv')).label.iloc[0],'Alice')
        self.assertIn('Integrity verification', (self.root/'report.md').read_text())

    def test_full_graph_rejects_generated_record_deletion(self):
        from app.agents import graph
        source=self.save('input.csv',pd.DataFrame({'record_id':['001','002','003'],'value':[10,20,30]}))
        generated=SimpleNamespace(code='def clean_data(df):\n    return df.iloc[1:]')
        with patch('app.agents.planner.get_settings',return_value=SimpleNamespace(GROQ_API_KEY='')),patch('app.agents.graph.generate_cleaning_code',return_value=generated):
            result=graph.run_pipeline('bad-job',source,str(self.root/'out.csv'),str(self.root/'report.md'))
        self.assertEqual(result['final_status'],'FAILED')
        self.assertFalse(result['validation_report']['integrity_passed'])
        self.assertEqual(result['retry_count'],3)

    def test_global_review_flag_does_not_block_independent_exact_deduplication(self):
        before=pd.DataFrame({'record_id':['001','001'],'value':[10,10]})
        after,review=apply_cleaning_plan(before,[action('flag_for_review'),action('drop_duplicates')])
        self.assertEqual(len(after),1)
        self.assertTrue(review)

    def test_timeout_cannot_publish_partial_output(self):
        source=self.save('input.csv',pd.DataFrame({'x':[1]}));out=str(self.root/'out.csv')
        result=execute_cleaning_code('def clean_data(df):\n    import time\n    time.sleep(2)\n    return df',source,out,timeout=0.05)
        self.assertFalse(result['success']);self.assertFalse(Path(out).exists())
        self.assertIn('timed out',result['error'])

    def test_supported_file_formats_preserve_text_tokens(self):
        frame=pd.DataFrame({'record_id':['001','002'],'country':['NA','US']})
        for extension in ['tsv','json','xlsx','parquet']:
            with self.subTest(format=extension):
                path=self.root/f'input.{extension}'
                if extension=='tsv':frame.to_csv(path,sep='\t',index=False)
                elif extension=='json':frame.to_json(path,orient='records')
                elif extension=='xlsx':frame.to_excel(path,index=False)
                else:frame.to_parquet(path,index=False)
                actual=load_dataframe(str(path))
                self.assertEqual(actual.record_id.tolist(),['001','002'])
                self.assertEqual(actual.country.tolist(),['NA','US'])

    def test_pipeline_review_preserves_ambiguous_amounts(self):
        from app.agents import graph
        source=self.save('review.csv',pd.DataFrame({'record_id':['001','002'],'amount':[-50.,100.]}))
        plan=TransformationPlan(dataset_summary='Uncertain amounts',actions=[TransformationAction(**action('flag_for_review','amount','low'))])
        with patch('app.agents.graph.generate_transformation_plan',return_value=plan),patch('app.agents.code_generator.get_settings',return_value=SimpleNamespace(GROQ_API_KEY='')):
            result=graph.run_pipeline('review-job',source,str(self.root/'review-out.csv'),str(self.root/'review.md'))
        self.assertEqual(result['final_status'],'FAILED')
        self.assertTrue(result['validation_report']['integrity_passed'])
        self.assertTrue(result['validation_report']['review_required'])
        self.assertFalse(result['validation_report']['overall_passed'])
        self.assertEqual(load_dataframe(str(self.root/'review-out.csv')).amount.tolist(),[-50.,100.])
        self.assertIn('NOT APPROVED', (self.root/'review.md').read_text())
