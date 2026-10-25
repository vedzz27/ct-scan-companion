"""Preview and re-ID scheduling regressions, without camera access."""
import os
os.environ['JOURNEY_LOG'] = ''
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import main
from test_matching import make_worker, frame_with
from store import Store

class CameraSpeedTests(unittest.TestCase):
    def test_live_preview_uses_latest_capture(self):
        w = make_worker(Store())
        w.grabber.frame, _ = frame_with([])
        w.grabber.t = time.time()
        # No detector frame or JPEG is available: preview must still work.
        with patch.dict(main.workers, {'test': w}):
            response = main.camera_image('test', live=True)
        self.assertEqual(response.media_type, 'image/jpeg')
        self.assertTrue(response.body.startswith(b'\xff\xd8'))

    def test_reid_is_throttled_but_position_is_updated(self):
        w = make_worker(Store())
        frame, people = frame_with([(200, 300, 400, 800, 1, 'red')])
        with patch.object(w.embedder, 'batch', wraps=w.embedder.batch) as batch:
            for t in (1000, 1000.1, 1000.2, 1000.3):
                w.process(frame, people, None, t)
            self.assertEqual(batch.call_count, 2)
            self.assertEqual(w.tracks[1].last_seen, 1000.3)

if __name__ == '__main__':
    unittest.main()
