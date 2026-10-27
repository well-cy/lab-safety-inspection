"""Regression tests for browser-compatible annotated videos."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from video.processor import process_video


class EmptyPipeline:
    cooldown_s = 10

    def analyze_frame(self, frame, areas, frame_time):
        return [], [], []


class VideoEncodingTests(unittest.TestCase):
    def test_output_is_h264_and_decodable(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'source.mp4', Path(tmp) / 'output.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.full((48, 64, 3), 127, dtype=np.uint8))
            writer.release()
            result = process_video(EmptyPipeline(), source, [], output, stride=1)
            self.assertEqual(result['total_frames'], 3)
            cap = cv2.VideoCapture(str(output))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.assertEqual(''.join(chr((fourcc >> (8*i)) & 255) for i in range(4)).lower(), 'h264')
            self.assertTrue(cap.read()[0])
            cap.release()
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['output.mp4', 'source.mp4'])

    def test_encoder_failure_removes_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'source.mp4', Path(tmp) / 'output.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            with patch('video.processor.subprocess.run', side_effect=OSError('encoder missing')):
                with self.assertRaisesRegex(RuntimeError, '视频转换失败'):
                    process_video(EmptyPipeline(), source, [], output, stride=1)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['source.mp4'])


if __name__ == '__main__':
    unittest.main()
