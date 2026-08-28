import unittest

import torch

from gaussian_renderer import (
    multiply_knn_by_splat_radius,
    normalize_knn_by_camera_distance,
)


class KnnCameraDistanceTest(unittest.TestCase):
    def test_divides_each_value_by_euclidean_camera_distance(self):
        values = torch.tensor([[1.0], [0.5]])
        xyz = torch.tensor([[0.0, 0.0, 2.0], [3.0, 0.0, 0.0]])
        camera_center = torch.zeros(3)

        normalized = normalize_knn_by_camera_distance(
            values,
            xyz,
            camera_center,
        )

        torch.testing.assert_close(normalized, torch.tensor([[0.5], [1.0 / 6.0]]))

    def test_camera_center_receives_distance_gradient(self):
        values = torch.tensor([[1.0], [0.5]])
        xyz = torch.tensor([[0.0, 0.0, 2.0], [3.0, 0.0, 0.0]])
        camera_center = torch.zeros(3, requires_grad=True)

        loss = normalize_knn_by_camera_distance(
            values,
            xyz,
            camera_center,
        ).sum()
        loss.backward()

        self.assertIsNotNone(camera_center.grad)
        self.assertTrue(torch.isfinite(camera_center.grad).all())
        self.assertGreater(float(camera_center.grad.norm()), 0.0)


class KnnSplatRadiusTest(unittest.TestCase):
    def test_multiplies_each_gaussian_by_cuda_pixel_radius(self):
        values = torch.tensor([[0.5], [0.25], [1.0]])
        radii = torch.tensor([4, 8, 0], dtype=torch.int32)

        weighted = multiply_knn_by_splat_radius(values, radii)

        torch.testing.assert_close(
            weighted,
            torch.tensor([[2.0], [2.0], [0.0]]),
        )

    def test_supports_three_channel_render_values(self):
        values = torch.tensor([[0.5, 0.5, 0.5]])
        radii = torch.tensor([6], dtype=torch.int32)

        weighted = multiply_knn_by_splat_radius(values, radii)

        torch.testing.assert_close(weighted, torch.full((1, 3), 3.0))


if __name__ == "__main__":
    unittest.main()
