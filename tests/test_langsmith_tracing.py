import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from app.agents import graph
from app.routers import jobs


class TraceQueryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = MagicMock()
        self.db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(id='job-a', created_at=datetime.now(timezone.utc))
        self.env = patch.dict(os.environ, {'LANGSMITH_API_KEY': 'test-key'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.query_patch = patch('app.langsmith_queries.fetch_job_runs', new_callable=AsyncMock)
        self.query = self.query_patch.start()
        self.addCleanup(self.query_patch.stop)

    async def test_keeps_event_contract_and_pending_state(self):
        now = datetime.now(timezone.utc)
        self.query.return_value = [SimpleNamespace(
            id='child', name='Validation Node', trace_id='trace-a', parent_run_ids=['root-a', 'parent-a'],
            start_time=now, end_time=None, error=None, tags=[], inputs={}, outputs={},
        )]
        events = await jobs.get_langsmith_traces('job-a', self.db)
        self.assertEqual(events[0]['parent_run_id'], 'parent-a')
        self.assertIsNone(events[0]['duration_ms'])
        self.assertEqual(events[0]['status'], 'pending')

    async def test_no_root_returns_empty(self):
        self.query.return_value = []
        self.assertEqual(await jobs.get_langsmith_traces('job-a', self.db), [])

    async def test_service_failure_is_visible(self):
        self.query.side_effect = RuntimeError('upstream unavailable')
        with self.assertLogs('app.routers.jobs', level='ERROR'):
            with self.assertRaises(HTTPException) as error:
                await jobs.get_langsmith_traces('job-a', self.db)
        self.assertEqual(error.exception.status_code, 502)

    async def test_missing_key_is_visible(self):
        with patch.dict(os.environ, {'LANGSMITH_API_KEY': '', 'LANGCHAIN_API_KEY': ''}):
            with self.assertRaises(HTTPException) as error:
                await jobs.get_langsmith_traces('job-a', self.db)
        self.assertEqual(error.exception.status_code, 503)

    async def test_unknown_job_is_404(self):
        self.db.query.return_value.filter.return_value.first.return_value = None
        with self.assertRaises(HTTPException) as error:
            await jobs.get_langsmith_traces('missing', self.db)
        self.assertEqual(error.exception.status_code, 404)


class TraceCompletionTests(unittest.TestCase):
    def test_flush_happens_after_root_returns_with_job_metadata(self):
        client = MagicMock()
        def run(**kwargs):
            client.flush.assert_not_called()
            self.assertEqual(kwargs['langsmith_extra']['metadata'], {'job_id': 'job-a'})
            return {'final_status': 'COMPLETED'}
        with patch('langsmith.run_trees.get_cached_client', return_value=client), patch.object(graph, '_run_pipeline', side_effect=run):
            self.assertEqual(graph.run_pipeline('job-a', 'in', 'out', 'report')['final_status'], 'COMPLETED')
        client.flush.assert_called_once_with(timeout=15)

    def test_flush_on_pipeline_failure_preserves_exception(self):
        client = MagicMock()
        with patch('langsmith.run_trees.get_cached_client', return_value=client), patch.object(graph, '_run_pipeline', side_effect=ValueError('pipeline failed')):
            with self.assertRaisesRegex(ValueError, 'pipeline failed'):
                graph.run_pipeline('job-a', 'in', 'out', 'report')
        client.flush.assert_called_once_with(timeout=15)

    def test_trace_flush_failure_does_not_fail_cleaning(self):
        client = MagicMock()
        client.flush.side_effect = RuntimeError('offline')
        with patch('langsmith.run_trees.get_cached_client', return_value=client), patch.object(graph, '_run_pipeline', return_value={'final_status': 'COMPLETED'}):
            with self.assertLogs('app.agents.graph', level='ERROR'):
                self.assertEqual(graph.run_pipeline('job-a', 'in', 'out', 'report')['final_status'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
