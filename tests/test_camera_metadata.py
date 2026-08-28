import unittest
from types import SimpleNamespace

import numpy as np

from scene.dataset_readers import CameraInfo
from utils.camera_utils import cameraList_from_camInfos


class CameraMetadataTest(unittest.TestCase):
    def test_builds_reference_camera_without_opening_image(self):
        camera_info = CameraInfo(
            uid=7,
            R=np.eye(3, dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
            FovY=0.7,
            FovX=0.9,
            depth_params=None,
            image_path="/path/that/does/not/exist.png",
            image_name="missing.png",
            depth_path="",
            width=3200,
            height=1600,
            is_test=False,
        )
        args = SimpleNamespace(resolution=-1)

        cameras = cameraList_from_camInfos(
            [camera_info],
            resolution_scale=1.0,
            args=args,
            is_nerf_synthetic=False,
            is_test_dataset=False,
            load_images=False,
        )

        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0].image_width, 1600)
        self.assertEqual(cameras[0].image_height, 800)
        self.assertEqual(cameras[0].image_name, "missing.png")
        np.testing.assert_array_equal(cameras[0].R, camera_info.R)
        np.testing.assert_array_equal(cameras[0].T, camera_info.T)


if __name__ == "__main__":
    unittest.main()
