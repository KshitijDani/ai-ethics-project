import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pandas as pd

from scripts import simulate
from scripts import simulate_single_turn


COLUMNS = ['model', 'mode', 'persona', 'question_file_name', 'question_number',
           'run_id', 'run_number', 'turn', 'question', 'user_message', 'response', 'timestamp']


class TopicSimulationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        # Create in reverse order; loading must sort filenames.
        (self.data / 'questions_food.txt').write_text('Food one\nFood two\n')
        (self.data / 'questions_book.txt').write_text('\ufeff  Book one  \n\nBook one\n')
        (self.data / 'questions.txt').write_text('Legacy input must be ignored')
        (self.data / 'question_old.txt').write_text('Old singular prefix must be ignored')
        (self.data / 'nested').mkdir()
        (self.data / 'nested/questions_ignored.txt').write_text('Ignored')
        (self.data / 'questions_directory.txt').mkdir()
        self.personas = self.root / 'Personas'
        self.personas.mkdir()
        for name in simulate.PERSONAS:
            (self.personas / f'{name}.txt').write_text(f'Persona {name}')
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'runs': 2, 'models': {
            name: {'base_url': 'http://localhost:8080/v1', 'api_model': name}
            for name in ('gemma', 'qwen', 'ministral')}}))
        self.output = self.root / 'output'
        self.payloads = []

    def request(self, url, timeout, payload=None):
        if payload is None:
            return {'data': [{'id': name} for name in ('gemma', 'qwen', 'ministral')]}
        self.payloads.append(copy.deepcopy(payload))
        return {'choices': [{'message': {'content': 'Answer, "quoted"\nline two',
                                         'reasoning_content': 'Private reasoning'},
                             'finish_reason': 'stop'}]}

    def invoke(self, module, *extra):
        stream = io.StringIO()
        with patch('scripts.simulate.request_json', side_effect=self.request), contextlib.redirect_stdout(stream):
            module.run(module.arguments([
                '--model', 'all', '--config', str(self.config),
                '--questions-folder', str(self.data), '--personas-folder', str(self.personas),
                '--output-folder', str(self.output), *extra]))
        return stream.getvalue()

    def test_matrix_order_csv_and_history_for_both_experiments(self):
        for module, experiment in ((simulate, 'multi-turn'), (simulate_single_turn, 'single-turn')):
            with self.subTest(experiment=experiment):
                self.payloads.clear()
                log = self.invoke(module)
                frames = [pd.read_csv(self.output / experiment / mode / f'{model}.csv')
                          for model in ('gemma', 'qwen', 'ministral') for mode in simulate.MODES]
                frame = pd.concat(frames, ignore_index=True)
                self.assertEqual(list(frame.columns), COLUMNS)
                self.assertEqual(len(frame), 168)
                self.assertEqual(len(self.payloads), 168)
                expected = []
                for model in ('gemma', 'qwen', 'ministral'):
                    for mode in simulate.MODES:
                        for topic in ('book', 'food'):
                            for persona in (('baseline',) if mode == 'baseline' else simulate.PERSONAS):
                                for repetition in (1, 2):
                                    for turn in (1, 2):
                                        expected.append((model, mode, topic, persona, repetition, turn))
                actual = []
                for row, payload in zip(frame.itertuples(), self.payloads):
                    topic = row.question_file_name.removeprefix('questions_').removesuffix('.txt')
                    actual.append((row.model, row.mode, topic, row.persona, row.run_number, row.turn))
                    self.assertEqual(row.question_number, f'{topic}_{row.turn}')
                    self.assertEqual(row.run_id,
                                     f'{experiment}/{row.model}/{row.mode}/{topic}/{row.persona}/{row.run_number}')
                    self.assertEqual(row.response, 'Answer, "quoted"\nline two')
                    self.assertEqual(payload['seed'], 41 + row.run_number)
                    personalized = row.mode != 'baseline' and (
                        experiment == 'single-turn' or row.mode == 'prompt-prepend' or row.turn == 1)
                    self.assertEqual(payload['messages'][-1]['content'].startswith('Persona '), personalized)
                    roles = [message['role'] for message in payload['messages']]
                    self.assertEqual(roles, ['system', 'user'] if experiment == 'single-turn' or row.turn == 1
                                     else ['system', 'user', 'assistant', 'user'])
                    if experiment == 'multi-turn' and row.turn == 2:
                        self.assertEqual(payload['messages'][2]['content'], row.response)
                self.assertEqual(actual, expected)
                self.assertEqual(frame.run_id.nunique(), 84)
                self.assertIn('Completed: 168', log)
                self.assertIn(str((self.output / experiment).resolve()), log)

    def test_default_and_verbose_logs(self):
        for module, experiment in ((simulate, 'multi-turn'), (simulate_single_turn, 'single-turn')):
            with self.subTest(experiment=experiment):
                log = self.invoke(module, '--mode', 'baseline', '--model', 'gemma', '--runs', '1')
                self.assertIn('Extracted 2 questions from questions_book.txt | topic=book | IDs=book_1..book_2', log)
                self.assertIn('Extracted 2 questions from questions_food.txt', log)
                self.assertIn('Loaded 4 questions from 2 files.', log)
                self.assertIn(f'Starting model=gemma | experiment={experiment}', log)
                self.assertIn('Starting mode=baseline | model=gemma', log)
                self.assertIn('Starting topic=book | file=questions_book.txt | questions=2', log)
                self.assertEqual(log.count('Starting run |'), 2)
                self.assertEqual(log.count('Completed run |'), 2)
                self.assertNotIn('Book one', log)
                self.assertNotIn('Starting request |', log)
                verbose_output = self.root / f'verbose-{experiment}'
                log = self.invoke(module, '--mode', 'prompt-prepend-once', '--model', 'gemma',
                                  '--runs', '1', '--verbose', '--output-folder', str(verbose_output))
                self.assertIn('Question | file=questions_book.txt | ID=book_1 | text=Book one', log)
                self.assertEqual(log.count('Question |'), 4)
                self.assertEqual(log.count('Starting request |'), 12)
                self.assertEqual(log.count('Completed request |'), 12)
                self.assertIn('question=book_1', log)
                self.assertNotIn('Persona American', log)
                self.assertNotIn('Private reasoning', log)
                self.assertNotIn('line two', log)

    def test_invalid_inputs_fail_before_server_start(self):
        for module in (simulate, simulate_single_turn):
            for filename, content, error in [('questions_.txt', 'One', 'topic'),
                                             ('questions_empty.txt', '\n ', 'No questions')]:
                path = self.data / filename
                path.write_text(content)
                with self.subTest(module=module.__name__, filename=filename), \
                        patch('start_server.managed_server') as start, \
                        patch('start_server.server_command', return_value=['fake']), \
                        patch('scripts.simulate.request_json') as request, contextlib.redirect_stdout(io.StringIO()):
                    args = module.arguments(['--model', 'all', '--config', str(self.config),
                                             '--questions-folder', str(self.data), '--manage-servers'])
                    with self.assertRaisesRegex(ValueError, error):
                        module.run(args)
                    start.assert_not_called()
                    request.assert_not_called()
                path.unlink()

    def test_missing_or_empty_directory_is_rejected(self):
        for folder in (self.root / 'missing', self.root / 'empty'):
            if folder.name == 'empty':
                folder.mkdir()
            with self.subTest(folder=folder), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'questions_.*topic'):
                    simulate.discover_questions(folder)

    def test_failure_has_context_and_no_false_completion(self):
        for module in (simulate, simulate_single_turn):
            for interrupt in (False, True):
                stream = io.StringIO()
                errors = io.StringIO()
                self.payloads.clear()
                def fail_second(url, timeout, payload=None):
                    if payload is not None and len(self.payloads) == 1:
                        if interrupt:
                            raise KeyboardInterrupt
                        raise RuntimeError('injected failure')
                    return self.request(url, timeout, payload)
                output = self.root / f'failed-{module.__name__}-{interrupt}'
                with patch('scripts.simulate.request_json', side_effect=fail_second), \
                        contextlib.redirect_stdout(stream), contextlib.redirect_stderr(errors):
                    code = module.main(['--model', 'gemma', '--mode', 'baseline', '--verbose',
                                        '--config', str(self.config), '--questions-folder', str(self.data),
                                        '--output-folder', str(output)])
                self.assertEqual(code, 130 if interrupt else 1)
                log = stream.getvalue()
                self.assertIn('question=book_2', log)
                self.assertEqual(log.count('Completed request |'), 1)
                self.assertNotIn('Completed run |', log)
                self.assertNotIn('Completed:', log)
                self.assertIn('Interrupted' if interrupt else 'Experiment did not complete', errors.getvalue())
                frame = pd.read_csv(next(output.rglob('*.csv')))
                self.assertEqual(frame.question_number.tolist(), ['book_1'])
                self.assertEqual(list(frame.columns), COLUMNS)

    def test_full_planned_matrix_counts(self):
        for path in self.data.glob('questions_*.txt'):
            if path.is_file():
                path.unlink()
        for topic in ('book', 'food', 'movie', 'hobbies'):
            (self.data / f'questions_{topic}.txt').write_text('\n'.join(f'{topic} {i}' for i in range(10)))
        for module, experiment in ((simulate, 'multi-turn'), (simulate_single_turn, 'single-turn')):
            self.payloads.clear()
            self.invoke(module, '--runs', '5')
            paths = list((self.output / experiment).rglob('*.csv'))
            self.assertEqual(len(paths), 9)
            self.assertEqual(sum(len(pd.read_csv(path)) for path in paths), 4200)
            self.assertEqual(len(self.payloads), 4200)

    def test_error_logs_do_not_dump_response_bodies(self):
        real_request = simulate.request_json
        marker = 'RESPONSE_BODY_MUST_NOT_BE_LOGGED'
        for module in (simulate, simulate_single_turn):
            for failure in ('http', 'malformed'):
                def request(url, timeout, payload=None):
                    if payload is None:
                        return {'data': [{'id': 'gemma'}]}
                    if failure == 'http':
                        return real_request(url, timeout, payload)
                    return {'choices': [{'message': {'content': None, 'reasoning_content': marker}}]}
                stream, errors = io.StringIO(), io.StringIO()
                http_error = HTTPError('http://localhost:8080/v1/chat/completions', 500,
                                       'error', {}, io.BytesIO(marker.encode()))
                self.addCleanup(http_error.close)
                with self.subTest(module=module.__name__, failure=failure), \
                        patch('scripts.simulate.request_json', side_effect=request), \
                        patch('scripts.simulate.urlopen', side_effect=http_error), \
                        contextlib.redirect_stdout(stream), contextlib.redirect_stderr(errors):
                    code = module.main(['--model', 'gemma', '--mode', 'baseline', '--verbose',
                                        '--config', str(self.config), '--questions-folder', str(self.data),
                                        '--output-folder', str(self.root / f'{module.__name__}-{failure}')])
                self.assertEqual(code, 1)
                self.assertNotIn(marker, stream.getvalue() + errors.getvalue())
                self.assertIn('HTTP 500' if failure == 'http' else 'Invalid chat response', errors.getvalue())
