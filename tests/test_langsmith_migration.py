"""Read-only SDK migration contract checks; no requests reach LangSmith."""
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from app.langsmith_queries import fetch_job_runs


class SmithDBQueryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patch = patch('app.langsmith_queries.Client', autospec=True)
        self.client = self.patch.start().return_value
        self.addCleanup(self.patch.stop)
        self.client.aread_project = AsyncMock(return_value=SimpleNamespace(id='project-uuid'))
        self.client.runs = MagicMock()
        self.client.traces = MagicMock()
        self.client.traces.list_runs = AsyncMock()
        self.created = datetime(2025, 1, 1)

    def roots(self, values):
        async def iterate():
            for value in values:
                yield value
        self.client.runs.query.return_value = iterate()

    async def fetch(self):
        return await fetch_job_runs('job-a', self.created, 'https://example.invalid', 'test-key', 'test-project')

    async def test_historical_job_fields_and_complete_trace(self):
        self.roots([SimpleNamespace(trace_id='trace-a')])
        items = [SimpleNamespace(id=str(i)) for i in range(150)]
        self.client.traces.list_runs.return_value = SimpleNamespace(items=items)
        self.assertEqual(await self.fetch(), items)
        query = self.client.runs.query.call_args.kwargs
        self.assertEqual(query['project_ids'], ['project-uuid'])
        self.assertTrue(query['is_root'])
        self.assertIn('"job-a"', query['filter'])
        self.assertEqual(query['min_start_time'], self.created.replace(tzinfo=timezone.utc) - timedelta(minutes=5))
        self.assertEqual(query['page_size'], 1)
        self.assertEqual(query['selects'], ['ID', 'TRACE_ID'])
        self.client.traces.list_runs.assert_awaited_once()
        args, kwargs = self.client.traces.list_runs.call_args
        self.assertEqual(args, ('trace-a',))
        self.assertEqual(kwargs['project_id'], 'project-uuid')
        self.assertNotIn('min_start_time', kwargs)
        self.assertTrue({'PARENT_RUN_IDS', 'INPUTS', 'OUTPUTS', 'START_TIME', 'END_TIME'} <= set(kwargs['selects']))
        self.client.close.assert_called_once()

    async def test_empty_root_does_not_fetch_some_other_trace(self):
        self.roots([])
        self.assertEqual(await self.fetch(), [])
        self.client.traces.list_runs.assert_not_awaited()
        self.client.close.assert_called_once()

    async def test_first_matching_root_stops_iteration(self):
        self.roots([SimpleNamespace(trace_id='first'), SimpleNamespace(trace_id='second')])
        self.client.traces.list_runs.return_value = SimpleNamespace(items=[])
        await self.fetch()
        self.assertEqual(self.client.traces.list_runs.call_args.args, ('first',))

    async def test_new_sdk_error_propagates_and_closes_client(self):
        from langsmith import APIConnectionError
        import httpx
        self.roots([SimpleNamespace(trace_id='trace-a')])
        error = APIConnectionError(request=httpx.Request('GET', 'https://example.invalid'))
        self.client.traces.list_runs.side_effect = error
        with self.assertRaises(APIConnectionError):
            await self.fetch()
        self.client.close.assert_called_once()


class ProjectDestinationTests(unittest.TestCase):
    def test_pipeline_names_project_when_agent_variables_are_also_set(self):
        from app.agents import graph
        from langsmith import get_current_run_tree, tracing_context
        client = MagicMock()
        observed = []
        fake_graph = MagicMock()
        def invoke(state):
            observed.append(get_current_run_tree().session_name)
            return {'final_status': 'COMPLETED'}
        fake_graph.invoke.side_effect = invoke
        with patch('app.agents.graph.build_graph', return_value=fake_graph), patch('langsmith.run_trees.get_cached_client', return_value=client):
            with tracing_context(enabled=True, client=client):
                graph.run_pipeline('diagnostic', 'in', 'out', 'report')
        self.assertEqual(observed, [graph.os.environ.get('LANGSMITH_PROJECT', 'agentic-data-preprocessing-pipeline')])
