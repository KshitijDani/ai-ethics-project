import json
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError
from pathlib import Path

import pandas as pd
import simulate_single_turn


class ManagedSingleTurnTests(unittest.TestCase):
    experiment = 'single-turn'
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.events = self.root / 'events'
        executable = self.root / 'fake-server'
        executable.write_text(f'''#!{Path(sys.executable).resolve()}
import json, signal, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
args = sys.argv
alias = args[args.index('--alias') + 1]
port = int(args[args.index('--port') + 1])
events = Path({str(self.events)!r})
def record(value):
    with events.open('a') as out: out.write(json.dumps(value) + '\\n')
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({{'data': [{{'id': alias}}]}}).encode())
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        record(payload)
        self.send_response(500 if 'fail' in payload['messages'][-1]['content'] else 200)
        self.end_headers()
        self.wfile.write(json.dumps({{'choices': [{{'message': {{'content': alias}}, 'finish_reason': 'stop'}}]}}).encode())
server = HTTPServer(('127.0.0.1', port), Handler)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
record('start ' + alias)
try: server.serve_forever()
finally:
    server.server_close()
    record('stop ' + alias)
''')
        executable.chmod(0o755)
        weights = self.root / 'test.gguf'
        weights.touch()
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'server_executable': str(executable), 'models': {
            name: {'model_path': str(weights), 'api_model': name,
                   'base_url': f'http://127.0.0.1:{self.port}/v1'}
            for name in ('gemma', 'qwen', 'ministral')}, 'runs': 1}))
        self.questions_folder = self.root / 'data'
        self.questions_folder.mkdir()
        self.questions = self.questions_folder / 'question_book.txt'
        self.questions.write_text('One\nTwo\n')
        self.output = self.root / 'output'
        self.csv_output = self.output / self.experiment

    def run_experiment(self, *extra):
        return simulate_single_turn.main([
            '--config', str(self.config), '--questions-folder', str(self.questions_folder),
            '--output-folder', str(self.output), '--mode', 'baseline',
            '--model', 'all', '--manage-servers', '--startup-timeout', '5', *extra])

    def test_sequential_servers_and_independent_requests(self):
        self.assertEqual(self.run_experiment(), 0)
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        self.assertEqual([event for event in events if isinstance(event, str)],
                         ['start gemma', 'stop gemma', 'start qwen', 'stop qwen',
                          'start ministral', 'stop ministral'])
        for payload in (event for event in events if isinstance(event, dict)):
            self.assertEqual([m['role'] for m in payload['messages']], ['system', 'user'])
        for name in ('gemma', 'qwen', 'ministral'):
            frame = pd.read_csv(self.csv_output / 'baseline' / f'{name}.csv')
            self.assertEqual(frame.response.tolist(), [name, name])

    def test_failure_stops_server_and_does_not_start_next(self):
        self.questions.write_text('One\nfail\n')
        self.assertEqual(self.run_experiment(), 1)
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        self.assertEqual([event for event in events if isinstance(event, str)],
                         ['start gemma', 'stop gemma'])
        self.assertEqual(len(pd.read_csv(self.csv_output / 'baseline/gemma.csv')), 1)

    def test_collision_on_later_model_prevents_any_start(self):
        target = self.csv_output / 'baseline/qwen.csv'
        target.parent.mkdir(parents=True)
        target.write_text('existing')
        self.assertEqual(self.run_experiment(), 1)
        self.assertFalse(self.events.exists())
        self.assertEqual(target.read_text(), 'existing')

    def test_busy_port_leaves_existing_server_alone(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', self.port))
            sock.listen()
            self.assertEqual(self.run_experiment(), 1)
            self.assertFalse(self.events.exists())

    def test_startup_timeout_stops_process(self):
        with patch('start_server.urlopen', side_effect=URLError('loading')):
            self.assertEqual(self.run_experiment('--startup-timeout', '1'), 1)
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        self.assertEqual(events, ['start gemma', 'stop gemma'])

    def test_early_process_exit_reports_failure(self):
        executable = self.root / 'fake-server'
        executable.write_text(f'#!{Path(sys.executable).resolve()}\nraise SystemExit(3)\n')
        self.assertEqual(self.run_experiment(), 1)
        self.assertFalse(self.events.exists())

    def test_interrupt_stops_server_and_preserves_completed_response(self):
        original = simulate_single_turn.request_answer
        calls = 0
        def answer(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return original(*args, **kwargs)
        with patch('simulate_single_turn.request_answer', side_effect=answer):
            self.assertEqual(self.run_experiment(), 130)
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        self.assertEqual([event for event in events if isinstance(event, str)],
                         ['start gemma', 'stop gemma'])
        self.assertEqual(len(pd.read_csv(self.csv_output / 'baseline/gemma.csv')), 1)


class ManagedMultiTurnTests(unittest.TestCase):
    experiment = 'multi-turn'
    setUp = ManagedSingleTurnTests.setUp
    test_failure_stops_server_and_does_not_start_next = ManagedSingleTurnTests.test_failure_stops_server_and_does_not_start_next
    test_collision_on_later_model_prevents_any_start = ManagedSingleTurnTests.test_collision_on_later_model_prevents_any_start
    test_startup_timeout_stops_process = ManagedSingleTurnTests.test_startup_timeout_stops_process

    def run_experiment(self, *extra):
        import simulate
        return simulate.main([
            '--config', str(self.config), '--questions-folder', str(self.questions_folder),
            '--output-folder', str(self.output), '--mode', 'baseline',
            '--model', 'all', '--manage-servers', '--startup-timeout', '5', *extra])

    def test_all_modes_retain_history_and_restart_conversations(self):
        import simulate
        personas = self.root / 'Personas'
        personas.mkdir()
        for name in simulate.PERSONAS:
            (personas / f'{name}.txt').write_text(f'I am {name}.')
        self.assertEqual(simulate.main([
            '--config', str(self.config), '--questions-folder', str(self.questions_folder),
            '--output-folder', str(self.output), '--personas-folder', str(personas),
            '--model', 'all', '--manage-servers', '--runs', '2']), 0)
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        self.assertEqual([event for event in events if isinstance(event, str)],
                         ['start gemma', 'stop gemma', 'start qwen', 'stop qwen',
                          'start ministral', 'stop ministral'])
        payloads = [event for event in events if isinstance(event, dict)]
        self.assertEqual(len(payloads), 84)
        for first, second in zip(payloads[::2], payloads[1::2]):
            self.assertEqual([m['role'] for m in first['messages']], ['system', 'user'])
            self.assertEqual([m['role'] for m in second['messages']],
                             ['system', 'user', 'assistant', 'user'])
            self.assertEqual(second['messages'][1], first['messages'][1])
        self.assertEqual(len(list(self.output.rglob('*.csv'))), 9)
        for name in ('gemma', 'qwen', 'ministral'):
            for mode, count in [('baseline', 4), ('prompt-prepend', 12), ('prompt-prepend-once', 12)]:
                frame = pd.read_csv(self.csv_output / mode / f'{name}.csv')
                self.assertEqual(len(frame), count)
                for _, conversation in frame.groupby('run_id', sort=False):
                    first, second = conversation.user_message.tolist()
                    if mode != 'baseline':
                        self.assertTrue(first.startswith('I am '))
                        self.assertEqual(second.startswith('I am '), mode == 'prompt-prepend')

    def test_interrupt_stops_server_and_preserves_completed_turn(self):
        import simulate
        original = simulate.conversation
        def conversation(*args, **kwargs):
            yield next(original(*args, **kwargs))
            raise KeyboardInterrupt
        with patch('simulate.conversation', side_effect=conversation):
            self.assertEqual(self.run_experiment(), 130)
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        self.assertEqual([event for event in events if isinstance(event, str)],
                         ['start gemma', 'stop gemma'])
        self.assertEqual(len(pd.read_csv(self.csv_output / 'baseline/gemma.csv')), 1)
