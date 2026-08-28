import unittest

import torch

from trajectory.losses import (
    apply_knn_threshold,
    knn_loss,
    roll_alignment_loss,
    rotation_acceleration_loss,
    tangent_alignment_loss,
)
from trajectory.model import _axis_angle_to_matrix


class RollAlignmentLossTest(unittest.TestCase):
    def setUp(self):
        self.reference_up = torch.tensor([[0.0, -1.0, 0.0]])

    def _rotation(self, axis_angle):
        return _axis_angle_to_matrix(torch.tensor([axis_angle], dtype=torch.float32))

    def test_does_not_penalize_pitch_or_yaw(self):
        pitch = self._rotation([0.4, 0.0, 0.0])
        yaw = self._rotation([0.0, 0.4, 0.0])

        self.assertAlmostEqual(
            float(roll_alignment_loss(pitch, self.reference_up)),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(roll_alignment_loss(yaw, self.reference_up)),
            0.0,
            places=6,
        )

    def test_penalizes_roll(self):
        roll = self._rotation([0.0, 0.0, 0.4])

        loss = roll_alignment_loss(roll, self.reference_up)

        self.assertGreater(float(loss), 0.0)
        self.assertAlmostEqual(float(loss), 1.0 - torch.cos(torch.tensor(0.4)), places=6)

    def test_view_direction_has_two_local_degrees_of_freedom(self):
        delta = torch.zeros(3, requires_grad=True)

        def forward_direction(value):
            return _axis_angle_to_matrix(value)[:, 2]

        jacobian = torch.autograd.functional.jacobian(forward_direction, delta)

        self.assertEqual(int(torch.linalg.matrix_rank(jacobian)), 2)


class RotationAccelerationLossTest(unittest.TestCase):
    def test_uniform_rotation_steps_have_zero_acceleration(self):
        axis_angles = torch.zeros(5, 3)
        axis_angles[:, 1] = torch.linspace(0.0, 0.8, 5)
        rotations = _axis_angle_to_matrix(axis_angles)

        loss = rotation_acceleration_loss(rotations)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_perturbed_waypoint_has_positive_acceleration(self):
        axis_angles = torch.zeros(5, 3)
        axis_angles[:, 1] = torch.linspace(0.0, 0.8, 5)
        rotations = _axis_angle_to_matrix(axis_angles)
        perturbation = _axis_angle_to_matrix(torch.tensor([0.3, 0.0, 0.0]))
        rotations[2] = rotations[2] @ perturbation

        loss = rotation_acceleration_loss(rotations)

        self.assertGreater(float(loss), 0.0)

    def test_short_path_has_zero_acceleration(self):
        rotations = torch.eye(3).repeat(2, 1, 1)
        self.assertEqual(float(rotation_acceleration_loss(rotations)), 0.0)


class KNNThresholdLossTest(unittest.TestCase):
    def test_values_below_absolute_threshold_share_one_cost(self):
        values = torch.tensor([[[0.01, 0.05, 0.2]]])

        thresholded = apply_knn_threshold(values, 0.05)

        torch.testing.assert_close(
            thresholded,
            torch.tensor([[[0.05, 0.05, 0.2]]]),
        )

    def test_values_below_threshold_have_zero_gradient(self):
        values = torch.tensor([[[0.01, 0.2]]], requires_grad=True)

        loss = knn_loss(values, threshold=0.05)
        loss.backward()

        torch.testing.assert_close(values.grad, torch.tensor([[[0.0, 0.5]]]))

    def test_background_maximum_cost_is_not_lowered(self):
        values = torch.tensor([[[1.0]]])
        self.assertEqual(float(apply_knn_threshold(values, 0.05)), 1.0)


class TangentAlignmentLossTest(unittest.TestCase):
    def test_uneven_collinear_segments_have_zero_loss(self):
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])

        self.assertAlmostEqual(float(tangent_alignment_loss(positions)), 0.0)

    def test_right_angle_has_unit_loss(self):
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        self.assertAlmostEqual(float(tangent_alignment_loss(positions)), 1.0)

    def test_zero_length_segment_is_ignored_safely(self):
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], requires_grad=True)

        loss = tangent_alignment_loss(positions)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(positions.grad).all())


if __name__ == "__main__":
    unittest.main()
