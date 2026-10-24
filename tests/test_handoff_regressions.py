"""交接核验回归：视频事件快照与输出目录迁移。"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import Config
from rule_engine.engine import ViolationEvent
from video.processor import process_video


class HandoffRegressions(unittest.TestCase):
    def test_output_root_moves_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith('LABSAFETY_')}
            env['LABSAFETY_OUTPUTS_DIR'] = tmp
            with patch.dict(os.environ, env, clear=True):
                c = Config()
                self.assertEqual(c.ANNOTATED_DIR, Path(tmp) / 'annotated')
                self.assertEqual(c.VIDEO_DIR, Path(tmp) / 'videos')
                self.assertEqual(c.SCREENSHOT_DIR, Path(tmp) / 'screenshots')
                with patch.dict(os.environ, {'LABSAFETY_VIDEO_DIR': tmp + '/custom'}):
                    self.assertEqual(Config().VIDEO_DIR, Path(tmp) / 'custom')

    def test_video_event_keeps_capture_time_and_screenshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = ViolationEvent(person_id=1, area_name='实验区',
                violation_types=['未佩戴口罩'], missing_ppe=['mask'],
                severity='一般违规', timestamp='2026-10-07 12:00:00')
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            capture = MagicMock()
            capture.isOpened.return_value = True
            capture.get.side_effect = [1.0, 32, 32, 4]
            capture.read.side_effect = [(True, frame)] * 4 + [(False, None)]
            pipeline = MagicMock()
            pipeline.screenshot_dir = Path(tmp)
            pipeline.cooldown_s = 2.0
            pipeline.analyze_frame.return_value = ([], [event], [])
            with patch('video.processor.cv2.VideoCapture', return_value=capture), \
                 patch('video.processor.draw_annotations', side_effect=lambda f, *a: f):
                result = process_video(pipeline, 'input.mp4', [], 'out.mp4',
                                       stride=2, save_annotated=False)
            self.assertEqual([e['frame_time'] for e in result['events']], [1.0, 3.0])
            self.assertNotEqual(result['events'][0]['screenshot'],
                                result['events'][1]['screenshot'])
            self.assertTrue(capture.release.called)


if __name__ == '__main__':
    unittest.main()
