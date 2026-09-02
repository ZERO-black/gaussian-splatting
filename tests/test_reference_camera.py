import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajectory.reference_camera import load_reference_cameras


class ReferenceCameraTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "camera.json"

    def tearDown(self):
        self.directory.cleanup()

    def _write(self, entries):
        self.path.write_text(json.dumps(entries))

    def test_loads_graphdeco_c2w_and_intrinsics(self):
        self._write(
            [
                {
                    "id": 7,
                    "img_name": "startup.jpg",
                    "width": 1920,
                    "height": 1080,
                    "position": [1, 2, 3],
                    "rotation": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                    "fx": 1080,
                    "fy": 1080,
                }
            ]
        )
        camera = load_reference_cameras(self.path)[0]

        np.testing.assert_allclose(camera.position, [1, 2, 3])
        np.testing.assert_allclose(camera.forward, [0, 1, 0])
        np.testing.assert_allclose(camera.up, [0, 0, 1])
        self.assertEqual(camera.image_width, 1920)
        self.assertEqual(camera.uid, 7)
        self.assertAlmostEqual(camera.FoVx, 2 * np.arctan(1920 / 2160))

    def test_rejects_non_rotation_matrix(self):
        self._write(
            [
                {
                    "width": 10,
                    "height": 10,
                    "position": [0, 0, 0],
                    "rotation": [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "fx": 5,
                    "fy": 5,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "not orthonormal"):
            load_reference_cameras(self.path)


if __name__ == "__main__":
    unittest.main()
