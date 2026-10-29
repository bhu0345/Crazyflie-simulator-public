import unittest

import numpy as np

import crazyflie_sim.model.params as params
from crazyflie_sim.model import linearization as nominal_model
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF


class TestHoverEquilibrium(unittest.TestCase):
    def test_u_eq_is_total_weight_thrust(self):
        _, u_eq = nominal_model.hover_equilibrium()
        self.assertTrue(
            np.array_equal(u_eq, np.array([CF.mass * params.g, 0.0, 0.0, 0.0]))
        )

    def test_x_eq_holds_pos_and_attitude(self):
        x_eq, _ = nominal_model.hover_equilibrium(
            pos=(1.0, -2.0, 3.0), attitude=(0.1, -0.2, 0.3)
        )
        self.assertTrue(np.array_equal(x_eq[0:3], np.array([1.0, -2.0, 3.0])))
        self.assertTrue(np.array_equal(x_eq[6:9], np.array([0.1, -0.2, 0.3])))
        self.assertTrue(np.array_equal(x_eq[3:6], np.zeros(3)))
        self.assertTrue(np.array_equal(x_eq[9:12], np.zeros(3)))

    def test_state_dot_vanishes_at_equilibrium(self):
        # With thrust = m*g along the body z axis, zero external force and zero
        # rates, every one of the 12 derivatives is ~0 — also for a translated
        # and yaw-rotated hover (the dynamics are position/yaw invariant).
        for pos, attitude in (
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((1.0, -2.0, 3.0), (0.0, 0.0, 0.3)),
        ):
            x_eq, u_eq = nominal_model.hover_equilibrium(pos=pos, attitude=attitude)
            state_dot = nominal_model._dynamics(x_eq, u_eq, np.zeros(3))
            self.assertLess(
                np.linalg.norm(state_dot),
                1e-9,
                "pos={} attitude={}".format(pos, attitude),
            )


class TestLinearizeHover(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A, cls.B, cls.x_eq, cls.u_eq = nominal_model.linearize_hover()

    def test_shapes_and_finiteness(self):
        self.assertEqual(self.A.shape, (12, 12))
        self.assertEqual(self.B.shape, (12, 4))
        self.assertTrue(np.isfinite(self.A).all())
        self.assertTrue(np.isfinite(self.B).all())

    def test_gravity_tilt_coupling(self):
        # RPYToRot is the world-to-body matrix, so the world-frame thrust is
        # F * (third row of R) = F*[cpsi*stheta+..., ..., cphi*ctheta]. Around
        # hover that gives d(x_ddot)/d(theta) = +F/m = +g and
        # d(y_ddot)/d(phi) = -F/m = -g; z_ddot is only second-order in tilt.
        g = params.g
        self.assertAlmostEqual(self.A[3, 7], +g, delta=0.01 * g)
        self.assertAlmostEqual(self.A[4, 6], -g, delta=0.01 * g)
        for col in (6, 7, 8):
            self.assertAlmostEqual(self.A[5, col], 0.0, places=12)

    def test_velocity_block_drag_jacobian_is_zero_at_hover(self):
        # Drag is quadratic (-factor*|v|*v), so its Jacobian at |v|=0 is
        # exactly 0. The tiny negative value the central difference reports
        # (-factor*h/m ~= -1.3e-7 at h=1e-6) is the O(h) truncation artifact
        # of the |v|*v term, not physics: assert small and non-positive.
        for idx in (3, 4, 5):
            self.assertLessEqual(self.A[idx, idx], 0.0)
            self.assertGreater(self.A[idx, idx], -1e-5)

    def test_angular_damping_diagonal(self):
        # pqr_dot = invI @ (M - omega x I omega - c_omega*omega), so the
        # diagonal of the rates block is -c_omega / I at hover.
        expected = -CF.angular_damping / np.diag(CF.inertia)
        for k, idx in enumerate((9, 10, 11)):
            self.assertLess(self.A[idx, idx], 0.0)
            self.assertAlmostEqual(
                self.A[idx, idx], expected[k], delta=0.01 * abs(expected[k])
            )

    def test_kinematics_and_euler_rate_identity(self):
        # d(pos)/dt = vel, and at zero attitude euler_rates is the identity
        # map (phi_dot = p, theta_dot = q, psi_dot = r).
        for pos_row, vel_col in ((0, 3), (1, 4), (2, 5)):
            self.assertAlmostEqual(self.A[pos_row, vel_col], 1.0, places=12)
        for att_row, rate_col in ((6, 9), (7, 10), (8, 11)):
            self.assertAlmostEqual(self.A[att_row, rate_col], 1.0, places=12)

    def test_thrust_input_column(self):
        # Column 0 (total thrust F) only affects z_ddot, with gain 1/mass.
        self.assertAlmostEqual(self.B[5, 0], 1.0 / CF.mass, delta=0.01 / CF.mass)
        for row in range(12):
            if row != 5:
                self.assertAlmostEqual(self.B[row, 0], 0.0, places=12)

    def test_moment_input_columns(self):
        # Columns 1..3 (moments) only affect the body-rate rows 9..11, with
        # gain invI (diagonal inertia).
        self.assertTrue(np.allclose(self.B[9:12, 1:4], np.linalg.inv(CF.inertia), rtol=0.01))
        self.assertLess(np.abs(self.B[0:9, 1:4]).max(), 1e-12)

    def test_controllability_rank(self):
        # ctrb = [B, AB, A^2 B, ..., A^11 B] must have full rank 12 at hover.
        ctrb = self.B.copy()
        for power in range(1, 12):
            ctrb = np.hstack([ctrb, np.linalg.matrix_power(self.A, power) @ self.B])
        self.assertEqual(ctrb.shape, (12, 48))
        self.assertEqual(np.linalg.matrix_rank(ctrb), 12)

    def test_matches_independent_finite_difference(self):
        # Spot-check the theta column (index 7) with an independent central
        # difference at a different step size than linearize() uses.
        h = 1e-5
        dx = np.zeros(12)
        dx[7] = h
        f_plus = nominal_model._dynamics(self.x_eq + dx, self.u_eq, np.zeros(3))
        f_minus = nominal_model._dynamics(self.x_eq - dx, self.u_eq, np.zeros(3))
        column = (f_plus - f_minus) / (2.0 * h)
        self.assertTrue(np.allclose(column, self.A[:, 7], rtol=1e-4, atol=1e-6))

    def test_deterministic_repeated_calls(self):
        A2, B2, _, _ = nominal_model.linearize_hover()
        self.assertTrue(np.array_equal(self.A, A2))
        self.assertTrue(np.array_equal(self.B, B2))


if __name__ == "__main__":
    unittest.main()
