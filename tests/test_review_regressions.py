"""Bugbot 回归：外置媒体目录、并发初始化、同秒任务排序。"""
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.tasks import TaskManager


class ReviewRegressions(unittest.TestCase):
    def test_external_media_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, LABSAFETY_OUTPUTS_DIR=tmp + '/outputs',
                       LABSAFETY_UPLOAD_DIR=tmp + '/uploads')
            for key in ('ANNOTATED', 'VIDEO', 'SCREENSHOT'):
                env['LABSAFETY_' + key + '_DIR'] = tmp + '/' + key.lower()
            code = '''
from backend.main import app
from backend.config import config
from fastapi.testclient import TestClient
client = TestClient(app)
for part, directory in [('annotated', config.ANNOTATED_DIR),
                        ('videos', config.VIDEO_DIR),
                        ('screenshots', config.SCREENSHOT_DIR)]:
    (directory / 'example.bin').write_bytes(part.encode())
    response = client.get('/media/' + part + '/example.bin')
    assert response.status_code == 200, (part, response.status_code)
    assert response.content == part.encode()
'''
            result = subprocess.run([sys.executable, '-B', '-c', code],
                cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_concurrent_first_load_constructs_one_model(self):
        import backend.main as main
        start = threading.Barrier(4)
        calls = []
        def construct(*args, **kwargs):
            calls.append(object())
            time.sleep(0.1)
            return calls[-1]
        def load(_):
            start.wait(timeout=5)
            return main.get_pipeline()
        with patch.object(main, '_pipeline', None), \
             patch.object(main, '_detector', None), \
             patch('ai.detector.PPEDetector', side_effect=construct), \
             patch('video.processor.InspectionPipeline', side_effect=lambda **kw: kw):
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(load, range(4)))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(result is results[0] for result in results))

    def test_latest_task_with_identical_display_time(self):
        hold = threading.Event()
        manager = TaskManager(max_workers=1)
        try:
            with patch('backend.tasks._now', return_value='2026-10-07 12:00:00'):
                ids = [manager.submit('test', lambda p: hold.wait(5))
                       for _ in range(3)]
                self.assertEqual([t.task_id for t in manager.list_recent(1)], ids[-1:])
                self.assertEqual([t.task_id for t in manager.list_recent(3)], ids[::-1])
        finally:
            hold.set()
            manager.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
