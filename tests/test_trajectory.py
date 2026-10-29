import unittest
import numpy as np
import math

from crazyflie_sim.sim import trajectory

class TestGetHelixWaypoints(unittest.TestCase):
    def test_get_helix_waypoints_size(self):
        t = 0
        n = 8
        waypoints = trajectory.get_helix_waypoints(t, n)
        expected_shape = (8, 4)
        self.assertEqual(waypoints.shape, expected_shape)

    def test_get_helix_waypoints_value(self):
        n = 8
        start_t = 0
        T = np.linspace(start_t, 2*np.pi, n)
        expected = np.zeros((n,4))

        for i, t in enumerate(T):
            expected[i] = np.array([0.5*np.cos(t), 0.5*np.sin(t), t, 0.0])

        waypoints = trajectory.get_helix_waypoints(start_t, n)
        np.testing.assert_array_equal(waypoints, expected)

class TestGetPolyCC(unittest.TestCase):
    def test_get_poly_cc_n(self):
        cc = trajectory.get_poly_cc(4, 0, 0)
        expected = [1, 0, 0, 0]
        np.testing.assert_array_equal(cc, expected)

        cc = trajectory.get_poly_cc(8, 0, 0)
        expected = [1, 0, 0, 0, 0, 0, 0, 0]
        np.testing.assert_array_equal(cc, expected)

    def test_get_poly_cc_k(self):
        cc = trajectory.get_poly_cc(4, 1, 1)
        expected = [0, 1, 2, 3]
        np.testing.assert_array_equal(cc, expected)

        cc = trajectory.get_poly_cc(4, 2, 2)
        expected = [0, 0, 2, 12]
        np.testing.assert_array_equal(cc, expected)

        cc = trajectory.get_poly_cc(8, 7, 1)
        expected = [0, 0, 0, 0, 0, 0, 0, math.factorial(7)]
        np.testing.assert_array_equal(cc, expected)

        cc = trajectory.get_poly_cc(8, 8, 1)
        expected = np.zeros(8)
        np.testing.assert_array_equal(cc, expected)

    def test_get_poly_cc_t(self):
        cc = trajectory.get_poly_cc(4, 0, 1)
        expected = [1, 1, 1, 1]
        np.testing.assert_array_equal(cc, expected)

        cc = trajectory.get_poly_cc(8, 0, 2)
        expected = [1, 2, 4, 8, 16, 32, 64, 128]
        np.testing.assert_array_equal(cc, expected)

        cc = trajectory.get_poly_cc(8, 1, 1)
        expected = np.linspace(0, 7, 8)
        np.testing.assert_array_equal(cc, expected)

        t = 2
        cc = trajectory.get_poly_cc(8, 1, t)
        expected = [0, 1, 2*t, 3*np.power(t,2), 4*np.power(t,3), 5*np.power(t,4), 6*np.power(t,5), 7*np.power(t,6)]
        np.testing.assert_array_equal(cc, expected)

    def test_get_poly_cc_input(self):
        with self.assertRaises(AssertionError):
            trajectory.get_poly_cc(-1, -1, 0)

        with self.assertRaises(AssertionError):
            trajectory.get_poly_cc(0, -3, -3)

class TestMST(unittest.TestCase):
    def test_invalid_segment_durations_are_rejected(self):
        for durations in ([1], [1, 0], [1, -1], [1, np.nan], [1, np.inf]):
            with self.subTest(durations=durations):
                with self.assertRaisesRegex(ValueError, "segment durations"):
                    trajectory.MST([0, 1, 3], durations)

    def test_MST_output_size(self):
        t = 0
        n = 8
        waypoints = trajectory.get_helix_waypoints(t, n)
        C = trajectory.MST(waypoints[:,0])
        self.assertEqual(C.shape, (8*(n-1), 1))

class TestGenerateTrajectory(unittest.TestCase):
    def test_unequal_segments_join_in_physical_time(self):
        waypoints = np.array([[0, 0, 1], [1, 0, 1], [3, 1, 2]], dtype=float)
        coeffs = trajectory.get_MST_coefficients(waypoints)
        for speed in (0.5, 1.0):
            with self.subTest(speed=speed):
                knot = 1.0 / speed
                before = trajectory.generate_trajectory(knot - 1e-8, speed, waypoints, *coeffs)
                after = trajectory.generate_trajectory(knot + 1e-8, speed, waypoints, *coeffs)
                np.testing.assert_allclose(before.pos, waypoints[1], atol=1e-6)
                np.testing.assert_allclose(after.pos, waypoints[1], atol=1e-6)
                np.testing.assert_allclose(before.vel, after.vel, atol=1e-6)
                np.testing.assert_allclose(before.acc, after.acc, atol=1e-6)
                h = 1e-6
                left = trajectory.generate_trajectory(knot - h, speed, waypoints, *coeffs)
                right = trajectory.generate_trajectory(knot + h, speed, waypoints, *coeffs)
                np.testing.assert_allclose((right.pos - left.pos) / (2 * h), after.vel, atol=1e-5)

    def test_unequal_segments_preserve_higher_derivatives(self):
        waypoints = np.array([[0, 0, 1], [1, 0, 1], [3, 1, 2]], dtype=float)
        durations = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        for coeffs in trajectory.get_MST_coefficients(waypoints):
            polynomials = [np.polynomial.Polynomial(c) for c in coeffs.reshape(-1, 8)]
            for order in range(1, 7):
                left = polynomials[0].deriv(order)(1) / durations[0] ** order
                right = polynomials[1].deriv(order)(0) / durations[1] ** order
                np.testing.assert_allclose(left, right, atol=1e-7)
            for order in range(1, 4):
                self.assertAlmostEqual(polynomials[0].deriv(order)(0), 0.0, places=7)
                self.assertAlmostEqual(polynomials[-1].deriv(order)(1), 0.0, places=7)

    def test_desired_state_size(self):
        t = 0
        n = 9
        v = 0.1 # m/s
        waypoints = trajectory.get_helix_waypoints(t, n)
        desired_state = trajectory.generate_trajectory(1.3, v, waypoints)
        self.assertEqual(desired_state.pos.shape, (3,))
        self.assertEqual(desired_state.vel.shape, (3,))
        self.assertEqual(desired_state.acc.shape, (3,))
        self.assertTrue(np.all(np.isfinite(desired_state.vel)))

if __name__ == '__main__':
    unittest.main()
