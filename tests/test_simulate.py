import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import simulate


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.requests = []
        self.fail_at = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'data': [{'id': m} for m in ('gemma', 'qwen', 'ministral')]}).encode())

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append(payload)
                if owner.fail_at == len(owner.requests):
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'injected failure')
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'choices': [{'message': {'content': 'Answer, "quoted"\nsecond line'}, 'finish_reason': 'stop'}]}).encode())

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        url = f'http://127.0.0.1:{self.server.server_port}/v1'
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'models': {m: {'base_url': url, 'api_model': m} for m in ('gemma', 'qwen', 'ministral')}, 'runs': 2}))
        self.questions = self.root / 'questions.txt'
        self.questions.write_text('Question one\n\nQuestion two\n')
        self.personas = self.root / 'Personas'
        self.personas.mkdir()
        for name in simulate.PERSONAS:
            (self.personas / f'{name}.txt').write_text(f'I am {name}.\nMy second paragraph.')
        self.output = self.root / 'Output Test'

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def args(self, *extra):
        return simulate.arguments(['--model', 'gemma', '--config', str(self.config), '--questions', str(self.questions), '--personas-folder', str(self.personas), '--output-folder', str(self.output), *extra])

    def run_args(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return simulate.run(self.args(*extra))

    def test_all_modes_counts_history_and_csv_roundtrip(self):
        self.run_args()
        self.assertEqual(len(list(self.output.rglob('*.csv'))), 3)
        baseline = pd.read_csv(self.output / 'baseline/gemma.csv')
        self.assertEqual(len(baseline), 4)
        for mode in simulate.MODES[1:]:
            for model in ('gemma',):
                frame = pd.read_csv(self.output / mode / f'{model}.csv')
                self.assertEqual(len(frame), 12)
                self.assertEqual(frame.run_id.nunique(), 6)
                for _, run in frame.groupby('run_id', sort=False):
                    self.assertTrue(run.iloc[0].user_message.startswith('I am '))
                    self.assertEqual(run.iloc[1].user_message.startswith('I am '), mode == 'prompt-prepend')
        self.assertEqual(baseline.iloc[0].response, 'Answer, "quoted"\nsecond line')
        self.assertEqual(len(self.requests), 28)
        for first, second in zip(self.requests[::2], self.requests[1::2]):
            self.assertEqual(len(first['messages']), 2)
            self.assertEqual(first['messages'][0], {'role': 'system', 'content': simulate.DEFAULT_SYSTEM_PROMPT})
            self.assertEqual([m['role'] for m in second['messages']], ['system', 'user', 'assistant', 'user'])
            self.assertEqual(second['messages'][0], first['messages'][0])
            self.assertEqual(second['messages'][1], first['messages'][1])
            self.assertEqual(second['messages'][2]['content'], baseline.iloc[0].response)

    def test_all_lines_duplicates_and_over_ten_questions(self):
        self.questions.write_text('\ufeff  same  \n\n' + 'same\n' * 11, encoding='utf-8')
        self.run_args('--mode', 'baseline', '--model', 'gemma', '--runs', '1')
        self.assertEqual(len(self.requests), 12)
        self.assertEqual(len(self.requests[-1]['messages']), 24)
        self.assertEqual(simulate.read_questions(self.questions), ['same'] * 12)

    def test_incremental_modes_and_collision_before_requests(self):
        self.run_args('--mode', 'baseline')
        original = (self.output / 'baseline/gemma.csv').read_bytes()
        self.run_args('--mode', 'prompt-prepend')
        self.assertEqual(original, (self.output / 'baseline/gemma.csv').read_bytes())
        calls = len(self.requests)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_args('--mode', 'baseline')
        self.assertEqual(calls, len(self.requests))

    def test_empty_input_and_wrong_model_fail_before_inference(self):
        self.questions.write_text(' \n')
        with self.assertRaisesRegex(ValueError, 'No questions'):
            self.run_args()
        self.questions.write_text('One\n')
        config = json.loads(self.config.read_text())
        config['models']['gemma']['api_model'] = 'wrong-model'
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'not served'):
            self.run_args()
        self.assertEqual(self.requests, [])

    def test_http_failure_saves_completed_turns_without_retry(self):
        self.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, 'HTTP 500'):
            self.run_args('--mode', 'baseline', '--model', 'gemma')
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(pd.read_csv(self.output / 'baseline/gemma.csv')), 1)

    def test_baseline_does_not_need_personas_and_override_needs_model(self):
        for path in self.personas.iterdir():
            path.unlink()
        self.run_args('--mode', 'baseline', '--model', 'gemma')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            simulate.arguments([])
        self.assertEqual(error.exception.code, 2)

    def test_sequential_models_share_folder_without_overwrite(self):
        originals = {}
        for model in ('gemma', 'qwen', 'ministral'):
            start = len(self.requests)
            self.run_args('--model', model)
            self.assertTrue(all(r['model'] == model for r in self.requests[start:]))
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)
            for mode in simulate.MODES:
                path = self.output / mode / f'{model}.csv'
                originals[path] = path.read_bytes()
        self.assertEqual(len(list(self.output.rglob('*.csv'))), 9)

    def test_unselected_endpoints_need_not_be_available(self):
        config = json.loads(self.config.read_text())
        config['models']['qwen']['base_url'] = 'http://127.0.0.1:1/v1'
        config['models']['ministral']['base_url'] = 'http://127.0.0.1:1/v1'
        self.config.write_text(json.dumps(config))
        self.run_args('--mode', 'baseline', '--model', 'gemma')
        self.assertEqual({r['model'] for r in self.requests}, {'gemma'})

    def test_custom_system_prompt_in_every_mode_and_csv(self):
        config = json.loads(self.config.read_text())
        config['system_prompt'] = 'Recommend exactly one item.\nKeep it brief.'
        self.config.write_text(json.dumps(config))
        self.run_args()
        for request in self.requests:
            self.assertEqual(request['messages'][0], {'role': 'system', 'content': config['system_prompt']})
            self.assertEqual(sum(m['role'] == 'system' for m in request['messages']), 1)
        for csv_path in self.output.rglob('*.csv'):
            self.assertEqual(set(pd.read_csv(csv_path).system_prompt), {config['system_prompt']})

    def test_empty_system_prompt_and_invalid_type(self):
        config = json.loads(self.config.read_text())
        config['system_prompt'] = ''
        self.config.write_text(json.dumps(config))
        self.run_args('--mode', 'baseline')
        self.assertTrue(all(m['role'] != 'system' for r in self.requests for m in r['messages']))
        config['system_prompt'] = None
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'system_prompt must be a string'):
            simulate.load_settings(self.args())


if __name__ == '__main__':
    unittest.main()
