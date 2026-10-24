"""BUG-008/009 API 回归；使用独立临时数据库和媒体目录。"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TEMP = tempfile.TemporaryDirectory(prefix='backend-008-009-')
ROOT = Path(TEMP.name)
for key, value in {'DB_PATH': ROOT / 'test.db', 'UPLOAD_DIR': ROOT / 'uploads',
                   'OUTPUTS_DIR': ROOT / 'outputs'}.items():
    os.environ['LABSAFETY_' + key] = str(value)
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend import main as api, database as db


class BackendBugs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)
        cls.client.__enter__()
        cls.videos = {}
        for suffix, codec in [('mp4', 'mp4v'), ('avi', 'MJPG')]:
            path = ROOT / ('valid.' + suffix)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 10, (64, 64))
            if not writer.isOpened():
                raise RuntimeError('测试环境无法生成真实视频')
            for _ in range(3):
                writer.write(np.zeros((64, 64, 3), np.uint8))
            writer.release()
            cls.videos[suffix] = path.read_bytes()

    @classmethod
    def tearDownClass(cls):
        api.task_manager.shutdown(wait=True)
        cls.client.__exit__(None, None, None)
        TEMP.cleanup()

    def lab(self):
        return db.create_lab('lab-' + str(time.time_ns()), 'original')

    def test_invalid_video_rejected_before_queue_or_model(self):
        png = cv2.imencode('.png', np.zeros((20, 20, 3), np.uint8))[1].tobytes()
        jpeg = cv2.imencode('.jpg', np.zeros((20, 20, 3), np.uint8))[1].tobytes()
        for endpoint in ['/api/detect/video', '/api/detect/video/async']:
            for payload in [png, jpeg, b'GIF89a' + b'\0'*50, b'not video', b'',
                            self.videos['mp4'][:32]]:
                with self.subTest(endpoint=endpoint, payload=payload[:8]):
                    before = set(api.UPLOAD_DIR.iterdir())
                    with patch.object(api, 'get_pipeline') as model, patch.object(api.task_manager, 'submit') as submit:
                        r = self.client.post(endpoint, files={'file': ('fake.mp4', payload, 'video/mp4')})
                    self.assertEqual(r.status_code, 400, r.text)
                    self.assertIn('error', r.json())
                    model.assert_not_called()
                    submit.assert_not_called()
                    self.assertEqual(set(api.UPLOAD_DIR.iterdir()), before)

    def test_real_video_sync_and_async(self):
        # 接口验收使用现有真实模型，避免假管线掩盖接口与落库问题。
        for suffix, payload in self.videos.items():
            for endpoint in ['/api/detect/video', '/api/detect/video/async']:
                with self.subTest(suffix=suffix, endpoint=endpoint):
                    r = self.client.post(endpoint, files={'file': ('real.' + suffix, payload, 'application/octet-stream')})
                    self.assertEqual(r.status_code, 202 if endpoint.endswith('async') else 200, r.text)
                    result = r.json()
                    if endpoint.endswith('async'):
                        deadline = time.monotonic() + 60
                        while time.monotonic() < deadline:
                            task = self.client.get(result['status_url']).json()
                            if task['status'] in ('done', 'failed'): break
                            time.sleep(.05)
                        self.assertEqual(task['status'], 'done', task)
                        result = task['result']
                    self.assertEqual(result['total_frames'], 3)
                    self.assertEqual(self.client.get(result['output_video_url']).status_code, 200)
                    with db.get_conn() as conn:
                        self.assertIsNotNone(conn.execute('SELECT id FROM detection_records WHERE id=?', (result['record_id'],)).fetchone())

    def test_edit_persists_unicode_and_keeps_references(self):
        lab = self.lab()
        area = db.create_area(lab, 'area', 0, 0, 1, 1, ['mask'])
        record = db.save_detection_record(lab, 'image', 'test', 1, 0, 1, 1)
        r = self.client.put(f'/api/settings/lab/{lab}', data={'name': '中文🧪', 'description': '第一行\n第二行'})
        self.assertEqual(r.status_code, 200)
        saved = next(x for x in self.client.get('/api/settings').json()['labs'] if x['id'] == lab)
        self.assertEqual(saved['name'], '中文🧪')
        self.assertEqual(saved['description'], '第一行\n第二行')
        self.assertEqual(saved['areas'][0]['id'], area)
        with db.get_conn() as conn:
            self.assertEqual(conn.execute('SELECT lab_id FROM detection_records WHERE id=?', (record,)).fetchone()[0], lab)

    def test_invalid_edit_leaves_original(self):
        lab = self.lab()
        for name, desc in [(' ', ''), ('a'*101, ''), ('valid', 'a'*501), ('bad\ufffd', ''), ('bad\x01', '')]:
            r = self.client.put(f'/api/settings/lab/{lab}', data={'name': name, 'description': desc})
            self.assertEqual(r.status_code, 400, r.text)
        with db.get_conn() as conn:
            self.assertEqual(conn.execute('SELECT description FROM laboratories WHERE id=?', (lab,)).fetchone()[0], 'original')

    def test_duplicate_and_unchanged_name(self):
        lab, other = self.lab(), self.lab()
        with db.get_conn() as conn:
            name = conn.execute('SELECT name FROM laboratories WHERE id=?', (lab,)).fetchone()[0]
        self.assertEqual(self.client.put(f'/api/settings/lab/{other}', data={'name': name}).status_code, 400)
        self.assertEqual(self.client.put(f'/api/settings/lab/{lab}', data={'name': name}).status_code, 200)

    def test_edit_length_boundaries(self):
        lab = self.lab()
        r = self.client.put(f'/api/settings/lab/{lab}', data={'name': '名'*100, 'description': '说'*500})
        self.assertEqual(r.status_code, 200, r.text)

    def test_single_frame_video_is_valid(self):
        from backend.validators import validate_video_file
        path = ROOT / 'single.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 64))
        self.assertTrue(writer.isOpened())
        writer.write(np.zeros((64, 64, 3), np.uint8))
        writer.release()
        validate_video_file(path)

    def test_delete_empty_and_missing(self):
        lab = self.lab()
        self.assertEqual(self.client.delete(f'/api/settings/lab/{lab}').status_code, 200)
        self.assertFalse(db.lab_exists(lab))
        self.assertEqual(self.client.delete(f'/api/settings/lab/{lab}').status_code, 404)
        self.assertEqual(self.client.put(f'/api/settings/lab/{lab}', data={'name':'missing'}).status_code, 404)

    def test_delete_rejects_each_reference(self):
        for kind in ('area', 'record', 'event'):
            with self.subTest(kind=kind):
                lab = self.lab()
                if kind == 'area': db.create_area(lab, 'area', 0, 0, 1, 1, ['mask'])
                elif kind == 'record': db.save_detection_record(lab, 'image', 'test', 0, 0, 0, 0)
                else:
                    with db.get_conn() as conn:
                        conn.execute("INSERT INTO violation_events(lab_id,area_name,person_id,violation_types,severity) VALUES(?,'area',1,'mask','minor')", (lab,))
                r = self.client.delete(f'/api/settings/lab/{lab}')
                self.assertEqual(r.status_code, 400, r.text)
                self.assertTrue(db.lab_exists(lab))
                with db.get_conn() as conn:
                    self.assertEqual(conn.execute('PRAGMA foreign_key_check').fetchall(), [])

if __name__ == '__main__':
    unittest.main()
