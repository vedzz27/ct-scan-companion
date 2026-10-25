"""Enrollment regressions without cameras or model downloads."""
import time
import unittest
from unittest.mock import patch
import numpy as np
from test_matching import ColourEmbedder, frame_with, make_worker, run
from store import Store
from worker import CameraWorker

class CheckinTests(unittest.TestCase):
    def worker(self):
        w = CameraWorker('kiosk', {'source': 0, 'role': 'kiosk'}, Store(), ColourEmbedder())
        w.frame, w.people = frame_with([(100, 100, 300, 600, 1, 'red')])
        w.processed_at = time.time()
        return w

    def test_readiness(self):
        w = self.worker()
        self.assertTrue(w.checkin_readiness()['ready'])
        w.processed_at -= 10
        self.assertFalse(w.checkin_readiness()['ready'])
        w.processed_at = time.time()
        w.people = [(*w.people[0][:4], -1, 0.9)]
        self.assertFalse(w.checkin_readiness()['ready'])

    def test_full_burst_and_single_capture(self):
        w = self.worker()
        def publish(_):
            w.people_seq += 1
            w.processed_at = time.time()
        with patch('worker.time.sleep', side_effect=publish):
            for n in (1, 6):
                crops, reason = w.capture_burst(n=n, timeout=0.1)
                self.assertIsNone(reason)
                self.assertEqual(len(crops), n)

    def test_old_frame_cannot_enroll(self):
        crops, _ = self.worker().capture_burst(n=1, timeout=0.02)
        self.assertIsNone(crops)

    def test_concurrent_capture_rejected(self):
        w = self.worker()
        with w.capture_lock:
            crops, reason = w.capture_burst()
        self.assertIsNone(crops)
        self.assertIn('progress', reason)

    def test_no_rematch_from_old_embeddings(self):
        s = Store()
        p = s.check_in('Red', [np.array([1, 0, 0, 0.1]) / np.sqrt(1.01)])
        w = make_worker(s)
        t = run(w, [(200, 300, 400, 800, 1, 'red')], 1000.)
        self.assertEqual(w.tracks[1].pid, p.id)
        last_eval = w.tracks[1].last_eval
        frame, people = frame_with([(200, 300, 400, 800, 1, 'red')])
        w.process(frame, [(*people[0][:5], 0.2)], None, t + 2)
        self.assertEqual(w.tracks[1].last_eval, last_eval)

if __name__ == '__main__':
    unittest.main()
