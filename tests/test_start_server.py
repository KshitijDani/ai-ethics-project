import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import start_server


class ServerConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()
        (self.folder / 'models').mkdir()
        self.model = self.folder / 'models/a model.gguf'
        self.model.touch()
        self.config = self.folder / 'settings.json'
        self.settings = {
            'server_executable': sys.executable,
            'models': {'qwen': {'model_path': 'models/a model.gguf',
                                'base_url': 'http://127.0.0.1:8081/v1', 'api_model': 'qwen',
                                'chat_template_kwargs': {'enable_thinking': False}}}
        }
        self.write()

    def write(self):
        self.config.write_text(json.dumps(self.settings))

    def test_config_relative_paths_and_spaces(self):
        cmd = start_server.server_command(self.config, 'qwen')
        self.assertEqual(cmd[cmd.index('--model') + 1], str(self.model))
        self.assertEqual(cmd[cmd.index('--port') + 1], '8081')
        self.assertEqual(json.loads(cmd[-1]), {'enable_thinking': False})

    def test_external_absolute_model_path(self):
        self.settings['models']['qwen']['model_path'] = str(self.model)
        self.write()
        self.assertIn(str(self.model), start_server.server_command(self.config, 'qwen'))

    def test_existing_endpoint_without_weights_is_simulator_only(self):
        del self.settings['models']['qwen']['model_path']
        self.write()
        with self.assertRaisesRegex(ValueError, 'model_path'):
            start_server.server_command(self.config, 'qwen')

    def test_missing_model_and_remote_endpoint_fail(self):
        self.model.unlink()
        with self.assertRaisesRegex(ValueError, 'not found'):
            start_server.server_command(self.config, 'qwen')
        self.model.touch()
        self.settings['models']['qwen']['base_url'] = 'http://example.org:8080/v1'
        self.write()
        with self.assertRaisesRegex(ValueError, 'local http'):
            start_server.server_command(self.config, 'qwen')

    def test_dry_run_does_not_start_process(self):
        with patch('start_server.subprocess.run') as run:
            self.assertEqual(start_server.main(['qwen', '--config', str(self.config), '--dry-run']), 0)
            run.assert_not_called()
