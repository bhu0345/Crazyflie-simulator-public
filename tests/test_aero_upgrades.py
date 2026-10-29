"""Tests for the aerodynamic realism upgrades (model/aero.py):

- rotor-induced drag (H-force), linear in the in-plane relative airflow
- body-frame anisotropic parasitic drag
- wind modeled as a velocity field (v_rel = v - v_wind); the force/velocity
  channel switch itself is covered in test_wind.py
- ground-effect thrust multiplier
- Quadcopter.body_frame() motor layout consistency with the mixer (the real
  Crazyflie numbering: M1 front-right, M2 rear-right, M3 rear-left, M4
  front-left)

Every feature is OFF by default; tests enable them explicitly and restore
module state in tearDown.
"""

import unittest

import numpy as np

import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.aero as aero
import crazyflie_sim.model.params as params
from crazyflie_sim.model.quadcopter import Quadcopter
from crazyflie_sim.model.sensors import IMU
from crazyflie_sim.utils.utils import RPYToRot
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF


class AeroTestCase(unittest.TestCase):
    """Save/restore every module flag these tests touch."""

    def setUp(self):
        self._saved = (
            params.rotor_drag_enabled,
            params.body_frame_drag_enabled,
            params.ground_effect_enabled,
        )
        params.rotor_drag_enabled = False
        params.body_frame_drag_enabled = False
        params.ground_effect_enabled = False

    def tearDown(self):
        (
            params.rotor_drag_enabled,
            params.body_frame_drag_enabled,
            params.ground_effect_enabled,
        ) = self._saved

    @staticmethod
    def _factors():
        return np.array(
            [
                params.aero_drag_factor_xy,
                params.aero_drag_factor_xy,
                params.aero_drag_factor_z,
            ]
        )


class TestLegacyBitIdentity(AeroTestCase):
    def test_default_matches_historical_inline_drag_bitwise(self):
        v = np.array([1.0, -2.0, 0.5])
        factors = self._factors()
        expected = -(factors * np.linalg.norm(v) * v)
        got = aero.aero_force_world(v, None, np.eye(3))
        self.assertTrue(np.array_equal(got, expected))

    def test_zero_wind_vector_is_bit_identical_to_none(self):
        v = np.array([0.3, -0.7, 0.2])
        got_none = aero.aero_force_world(v, None, np.eye(3))
        got_zero = aero.aero_force_world(v, np.zeros(3), np.eye(3))
        self.assertTrue(np.array_equal(got_none, got_zero))


class TestRotorDrag(AeroTestCase):
    def setUp(self):
        super().setUp()
        params.rotor_drag_enabled = True

    def test_opposes_inplane_motion_linearly(self):
        v = np.array([1.0, 0.0, 0.0])
        force = aero.aero_force_world(v, None, np.eye(3))
        kxy = params.aero_drag_factor_xy
        dx = params.rotor_drag_coefficient_x
        # Legacy world quadratic PLUS rotor drag: |v| = 1 -> -kxy*1*1 in x.
        np.testing.assert_allclose(force, [-(kxy + dx), 0.0, 0.0], atol=1e-15)
        # Linear in v: doubling the speed doubles the ROTOR part only.
        force2 = aero.aero_force_world(2.0 * v, None, np.eye(3))
        rotor_part_1 = force[0] + kxy * 1.0 * 1.0
        rotor_part_2 = force2[0] + kxy * 2.0 * 2.0
        self.assertAlmostEqual(rotor_part_2, 2.0 * rotor_part_1, places=15)

    def test_no_z_component(self):
        v = np.array([0.0, 0.0, 1.0])
        force = aero.aero_force_world(v, None, np.eye(3))
        # Only the legacy parasitic z drag; rotor drag acts in-plane only.
        self.assertAlmostEqual(force[2], -params.aero_drag_factor_z, places=15)

    def test_follows_the_rotor_plane_when_tilted(self):
        phi, theta, psi = 0.0, np.pi / 6.0, 0.0
        bRw = RPYToRot(phi, theta, psi)
        # Construct a world velocity that is purely along body x.
        v_rel_body = np.array([1.0, 0.0, 0.0])
        v_world = bRw.T.dot(v_rel_body)
        force = aero.aero_force_world(v_world, None, bRw)
        expected = -(
            self._factors() * np.linalg.norm(v_world) * v_world
        ) + bRw.T.dot(
            np.array([-params.rotor_drag_coefficient_x, 0.0, 0.0])
        )
        np.testing.assert_allclose(force, expected, atol=1e-15)
        # Sanity: the rotor-drag part is NOT aligned with the world x axis
        # (it lies in the tilted rotor plane), unlike the legacy model.
        rotor_world = bRw.T.dot(np.array([-params.rotor_drag_coefficient_x, 0.0, 0.0]))
        self.assertNotAlmostEqual(rotor_world[2], 0.0, places=6)


class TestBodyFrameDrag(AeroTestCase):
    def setUp(self):
        super().setUp()
        params.body_frame_drag_enabled = True

    def test_componentwise_quadratic_not_norm_coupled(self):
        v = np.array([0.6, 0.8, 0.0])  # |v| = 1
        force = aero.aero_force_world(v, None, np.eye(3))
        kxy = params.aero_drag_factor_xy
        np.testing.assert_allclose(
            force, [-kxy * 0.36, -kxy * 0.64, 0.0], atol=1e-15
        )
        # The legacy norm-coupled model would give -kxy * [0.6, 0.8].
        self.assertNotAlmostEqual(force[0], -kxy * 0.6, places=15)

    def test_anisotropy_rotates_with_the_body(self):
        # Yawed 90 deg: world-x motion is body-(-y) motion. With equal xy
        # factors the magnitude is unchanged, but a world-frame model and a
        # body-frame model must agree here -- so use pitch to discriminate:
        # world-z motion acquires a body-x component when pitched.
        theta = np.pi / 4.0
        bRw = RPYToRot(0.0, theta, 0.0)
        v = np.array([0.0, 0.0, 1.0])
        v_body = bRw.dot(v)
        expected = bRw.T.dot(-self._factors() * np.abs(v_body) * v_body)
        force = aero.aero_force_world(v, None, bRw)
        np.testing.assert_allclose(force, expected, atol=1e-15)


class TestWindVelocityField(AeroTestCase):
    def test_stationary_quad_in_wind_equals_moving_quad_in_calm_air(self):
        params.rotor_drag_enabled = True
        params.body_frame_drag_enabled = True
        wind = np.array([2.0, -1.0, 0.5])
        f_wind = aero.aero_force_world(np.zeros(3), wind, np.eye(3))
        f_calm = aero.aero_force_world(-wind, None, np.eye(3))
        self.assertTrue(np.array_equal(f_wind, f_calm))

    def test_wind_pushes_downwind(self):
        params.rotor_drag_enabled = True
        force = aero.aero_force_world(
            np.zeros(3), np.array([2.0, 0.0, 0.0]), np.eye(3)
        )
        self.assertGreater(force[0], 0.0)

    def test_plant_state_dot_feels_wind(self):
        params.rotor_drag_enabled = True
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        F = CF.mass * params.g
        M = np.zeros((3, 1))
        sd_calm = quad.state_dot(quad.state, 0.0, F, M, np.zeros(3), None)
        np.testing.assert_allclose(sd_calm[3:6], np.zeros(3), atol=1e-12)
        sd_wind = quad.state_dot(
            quad.state, 0.0, F, M, np.zeros(3), np.array([2.0, 0.0, 0.0])
        )
        self.assertGreater(sd_wind[3], 0.0)


class TestGroundEffect(AeroTestCase):
    def test_disabled_returns_exactly_one(self):
        for z in (0.0, 0.01, 0.1, 1.0):
            self.assertEqual(aero.ground_effect_factor(z), 1.0)

    def test_factor_values(self):
        params.ground_effect_enabled = True
        r = params.prop_radius
        # Far away: essentially unity.
        self.assertAlmostEqual(
            aero.ground_effect_factor(1.0), 1.0 / (1.0 - (r / 4.0) ** 2), places=12
        )
        # z = R: 1/(1 - (1/4)^2) = 16/15.
        self.assertAlmostEqual(aero.ground_effect_factor(r), 16.0 / 15.0, places=12)
        # z -> 0 clamps at z = R/2: factor saturates at 4/3 (no singularity).
        self.assertAlmostEqual(aero.ground_effect_factor(0.0), 4.0 / 3.0, places=12)
        # Monotone decreasing with height.
        zs = np.linspace(0.0, 0.3, 20)
        factors = [aero.ground_effect_factor(z) for z in zs]
        self.assertTrue(all(b <= a + 1e-15 for a, b in zip(factors, factors[1:])))

    def test_update_applies_thrust_multiplier(self):
        params.ground_effect_enabled = True
        hover_pwm = actuators.wrench_to_pwm(
            CF.mass * params.g,
            np.zeros((3, 1), dtype=float),
            params.battery_voltage_nominal,
        )
        quad_ge = Quadcopter((0.0, 0.0, 0.02), (0.0, 0.0, 0.0))
        quad_plain = Quadcopter((0.0, 0.0, 0.02), (0.0, 0.0, 0.0))
        quad_ge.update(0.002, hover_pwm, external_force_world=np.zeros(3))
        params.ground_effect_enabled = False
        quad_plain.update(0.002, hover_pwm, external_force_world=np.zeros(3))
        params.ground_effect_enabled = True

        self.assertGreater(quad_ge.ground_effect_factor, 1.0)
        self.assertEqual(quad_plain.ground_effect_factor, 1.0)
        # Same command, more realized thrust near the ground -> climbs sooner.
        self.assertGreater(quad_ge.position()[2], quad_plain.position()[2])

    def test_accelerometer_sees_ground_effect_thrust(self):
        params.ground_effect_enabled = True
        imu = IMU(
            gyro_noise_std=0.0,
            gyro_bias_std=0.0,
            accel_noise_std=0.0,
            accel_bias_std=0.0,
            rng=np.random.default_rng(0),
        )
        hover_pwm = actuators.wrench_to_pwm(
            CF.mass * params.g,
            np.zeros((3, 1), dtype=float),
            params.battery_voltage_nominal,
        )
        quad = Quadcopter((0.0, 0.0, 0.02), (0.0, 0.0, 0.0))
        quad.update(0.002, hover_pwm, external_force_world=np.zeros(3))
        accel_meas = imu.accel(quad, np.zeros(3))
        # Specific force z = realized thrust / m > g when GE augments thrust.
        self.assertGreater(accel_meas[2], params.g)


class TestBodyFrameLayout(AeroTestCase):
    def test_motor_positions_match_mixer_wrench(self):
        # A unit thrust at motor position (x, y) produces body torque
        # r x F = (y*f, -x*f, 0); the mixer's Mx/My must agree with
        # the drawn body frame.
        frame = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)).body_frame()
        for i in range(4):
            e = np.zeros(4)
            e[i] = 1.0
            _, M = actuators.motor_forces_to_wrench(e)
            M = np.asarray(M, dtype=float).reshape(3,)
            x, y = frame[i, 0], frame[i, 1]
            self.assertAlmostEqual(float(M[0]), y, places=12, msg="motor {}".format(i + 1))
            self.assertAlmostEqual(float(M[1]), -x, places=12, msg="motor {}".format(i + 1))

    def test_real_crazyflie_numbering_and_spin(self):
        # M1 front-right, M2 rear-right, M3 rear-left, M4 front-left
        # (+x forward, +y left).
        signs = np.sign(Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)).body_frame()[:4, :2])
        np.testing.assert_array_equal(
            signs,
            np.array(
                [
                    [1.0, -1.0],  # m1 front-right
                    [-1.0, -1.0],  # m2 rear-right
                    [-1.0, 1.0],  # m3 rear-left
                    [1.0, 1.0],  # m4 front-left
                ]
            ),
        )
        # CCW props (m1, m3) create negative (CW) reaction torque on the body,
        # CW props (m2, m4) positive -- matching the real Crazyflie 2.x.
        mz_signs = []
        for i in range(4):
            e = np.zeros(4)
            e[i] = 1.0
            _, M = actuators.motor_forces_to_wrench(e)
            M = np.asarray(M, dtype=float).reshape(3,)
            mz_signs.append(np.sign(float(M[2])))
        self.assertEqual(mz_signs, [-1.0, 1.0, -1.0, 1.0])

    def test_pg_frame_rotation_matches_layout(self):
        try:
            from crazyflie_sim.visualization.quad_plot_pg import _frame_rotation_and_origin
        except ImportError:
            self.skipTest("pyqtgraph not installed")
        quad = Quadcopter((0.1, 0.2, 0.3), (0.2, -0.1, 0.7))
        rot, origin = _frame_rotation_and_origin(quad.world_frame())
        np.testing.assert_allclose(rot, quad.rotation_body_to_world(), atol=1e-9)
        np.testing.assert_allclose(origin, quad.position(), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
