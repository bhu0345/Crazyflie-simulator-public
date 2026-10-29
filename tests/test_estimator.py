import unittest

import numpy as np

from crazyflie_sim.sim.scenario import DT
from crazyflie_sim.sim.simulation import Simulation
from crazyflie_sim.estimation.estimator import ComplementaryEstimator, build_estimator
from crazyflie_sim.model.lighthouse import (
    LH2_ANGLE_BIAS_STD,
    LH2_ANGLE_NOISE_STD,
    LH2_DELAY_STEPS,
    LH2_PROTOCOL,
    LH2_RATE_HZ,
    LH2_TILT_RAD,
    Lighthouse,
    default_basestations,
    sweep_angle_and_grad,
)
from crazyflie_sim.estimation.nav_ekf import ErrorStateNavEKF
from crazyflie_sim.model.quadcopter import Quadcopter
from crazyflie_sim.model.sensors import IMU, Mocap
from crazyflie_sim.utils.utils import RPYToRot


ZERO_NOISE_CFG = {
    "enabled": True,
    "backend": "complementary",
    "imu": {
        "gyro_noise_std": 0.0,
        "gyro_bias_std": 0.0,
        "accel_noise_std": 0.0,
        "accel_bias_std": 0.0,
    },
    "mocap": {
        "rate_hz": 500.0,  # >= control rate: correction every step
        "pos_noise_std": 0.0,
        "att_noise_std": 0.0,
        "delay_steps": 0,
    },
    "gains": {"k_pos": 0.4, "k_vel": 0.1, "k_att": 0.3, "k_bias": 0.0},
}

NOISY_CFG = {
    "enabled": True,
    "backend": "complementary",
    "imu": {
        "gyro_noise_std": 0.005,
        "gyro_bias_std": 0.02,
        "accel_noise_std": 0.1,
        "accel_bias_std": 0.05,
    },
    "mocap": {
        "rate_hz": 100.0,
        "pos_noise_std": 0.002,
        "att_noise_std": 0.01,
        "delay_steps": 2,
    },
    "gains": {"k_pos": 0.4, "k_vel": 0.1, "k_att": 0.3, "k_bias": 0.01},
}


class TestZeroNoiseTracking(unittest.TestCase):
    """With all noise off and mocap at the control rate, the estimate must
    track the truth tightly over a short hover run."""

    def test_estimate_tracks_truth(self):
        run = Simulation("baseline", duration_s=2.0, estimator=(ZERO_NOISE_CFG, 42)).run()
        mask = run["t"] >= 1.0
        pos_err = np.linalg.norm(
            run["est_pos"][mask] - run["pos"][mask], axis=1
        )
        att_err = np.linalg.norm(
            run["est_att"][mask] - run["att"][mask], axis=1
        )
        self.assertLess(float(pos_err.max()), 5e-3)  # within a few mm
        self.assertLess(float(att_err.max()), 1e-3)
        # Sanity: closing the loop on the (near-perfect) estimate still hovers.
        self.assertLess(float(run["err"][-1]), 0.05)

    def test_constant_gyro_bias_converges_with_default_learning_gain(self):
        import copy

        cfg = copy.deepcopy(ZERO_NOISE_CFG)
        cfg.pop("gains")
        cfg["mocap"]["rate_hz"] = 100.0
        for bias in (np.array([0.02, -0.015, 0.01]), np.array([-0.02, 0.015, -0.01])):
            with self.subTest(bias=bias):
                quad = Quadcopter((0, 0, 1), (0, 0, 0))
                est = build_estimator(cfg, quad, DT, seed=0)
                est.imu.gyro_bias[:] = bias
                for k in range(int(50.0 / DT)):
                    est.update(quad, DT, (k + 1) * DT, np.zeros(3))
                self.assertTrue(np.all(est.gyro_bias_est * bias > 0))
                self.assertLess(np.linalg.norm(est.gyro_bias_est - bias), 0.25 * np.linalg.norm(bias))
                self.assertLess(np.linalg.norm(est.view(quad).omega()), 0.25 * np.linalg.norm(bias))


class TestSeedReproducibility(unittest.TestCase):
    def test_same_seed_identical_measurement_sequences(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        kwargs = dict(
            gyro_noise_std=0.01,
            gyro_bias_std=0.02,
            accel_noise_std=0.1,
            accel_bias_std=0.05,
        )
        imu_a = IMU(rng=np.random.default_rng(7), **kwargs)
        imu_b = IMU(rng=np.random.default_rng(7), **kwargs)
        imu_c = IMU(rng=np.random.default_rng(8), **kwargs)

        gyro_a = np.array([imu_a.gyro(quad.omega()) for _ in range(20)])
        gyro_b = np.array([imu_b.gyro(quad.omega()) for _ in range(20)])
        gyro_c = np.array([imu_c.gyro(quad.omega()) for _ in range(20)])
        np.testing.assert_array_equal(gyro_a, gyro_b)
        self.assertFalse(np.array_equal(gyro_a, gyro_c))

        accel_a = np.array([imu_a.accel(quad, np.zeros(3)) for _ in range(20)])
        accel_c = np.array([imu_c.accel(quad, np.zeros(3)) for _ in range(20)])
        self.assertFalse(np.array_equal(accel_a, accel_c))

    def test_same_seed_identical_estimates_across_runs(self):
        def drive(seed):
            quad = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            est = build_estimator(NOISY_CFG, quad, DT, seed=seed)
            pwm = quad.motor_pwm.copy()  # hover PWM, no controller needed
            hist = []
            for k in range(250):  # 0.5 s
                t = float(k + 1) * DT
                quad.update(DT, pwm, external_force_world=np.zeros(3))
                est.update(quad, DT, t, np.zeros(3))
                hist.append(
                    np.concatenate(
                        [est.pos_est, est.vel_est, est.att_est, est.view(quad).omega()]
                    )
                )
            return np.array(hist)

        hist_a = drive(7)
        hist_b = drive(7)
        hist_c = drive(8)
        np.testing.assert_array_equal(hist_a, hist_b)
        self.assertFalse(np.array_equal(hist_a, hist_c))


class TestEstimatedQuadView(unittest.TestCase):
    def test_state_accessors_return_estimate_rest_delegates_to_truth(self):
        quad = Quadcopter((1.0, 2.0, 3.0), (0.1, 0.2, 0.3))
        est = build_estimator(NOISY_CFG, quad, DT, seed=1)
        est.pos_est = np.array([9.0, 8.0, 7.0])
        est.vel_est = np.array([0.3, 0.2, 0.1])
        est.att_est = np.array([0.5, -0.4, 0.3])

        view = est.view(quad)
        np.testing.assert_array_equal(view.position(), est.pos_est)
        np.testing.assert_array_equal(view.velocity(), est.vel_est)
        np.testing.assert_array_equal(view.attitude(), est.att_est)
        np.testing.assert_array_equal(view.omega(), est._omega_est)
        np.testing.assert_array_equal(
            view.rotation_world_to_body(), RPYToRot(0.5, -0.4, 0.3)
        )
        np.testing.assert_array_equal(
            view.rotation_body_to_world(), RPYToRot(0.5, -0.4, 0.3).T
        )
        # The view must NOT show the truth through the state accessors.
        self.assertFalse(np.array_equal(view.position(), quad.position()))

        # Everything else resolves to the real quad.
        self.assertEqual(view.battery_voltage, quad.battery_voltage)
        self.assertIs(view.rotor_fault, quad.rotor_fault)
        self.assertIs(view.state, quad.state)
        np.testing.assert_array_equal(view.motor_thrusts, quad.motor_thrusts)
        # Plot helpers intentionally show the truth.
        np.testing.assert_array_equal(view.world_frame(), quad.world_frame())


class TestMocapTiming(unittest.TestCase):
    def test_rate_gating_and_delay_ring_buffer(self):
        quad = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        mocap = Mocap(
            rate_hz=100.0,
            pos_noise_std=0.0,
            att_noise_std=0.0,
            delay_steps=2,
            rng=np.random.default_rng(0),
        )
        # Emits at t = 0; before the delay buffer fills, serves the oldest.
        pos, _ = mocap.update(quad, 0.0)
        np.testing.assert_array_equal(pos, [0.0, 0.0, 0.0])
        # Between multiples: no sample.
        self.assertIsNone(mocap.update(quad, 0.005))
        # Clearly before the next multiple: still no sample.
        self.assertIsNone(mocap.update(quad, 0.009999))
        # A last-ulp-style near miss of the multiple still emits.
        quad.state[0:3] = [1.0, 0.0, 0.0]
        pos, _ = mocap.update(quad, 0.01 - 1e-12)
        np.testing.assert_array_equal(pos, [0.0, 0.0, 0.0])  # delayed: oldest
        # ...and the multiple is not emitted twice.
        self.assertIsNone(mocap.update(quad, 0.01 + 1e-12))
        quad.state[0:3] = [2.0, 0.0, 0.0]
        pos, _ = mocap.update(quad, 0.02)
        np.testing.assert_array_equal(pos, [0.0, 0.0, 0.0])  # buffer now full
        quad.state[0:3] = [3.0, 0.0, 0.0]
        pos, _ = mocap.update(quad, 0.03)
        np.testing.assert_array_equal(pos, [1.0, 0.0, 0.0])  # 2 emissions ago


class TestBuildEstimatorValidation(unittest.TestCase):
    def setUp(self):
        self.quad = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_missing_enabled_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_estimator({"imu": {}}, self.quad, DT, seed=0)
        self.assertIn("enabled", str(ctx.exception))

    def test_unknown_keys_raise_and_name_the_key(self):
        with self.assertRaises(ValueError) as ctx:
            build_estimator({"enabled": True, "gps": {}}, self.quad, DT, seed=0)
        self.assertIn("gps", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            build_estimator(
                {"enabled": True, "mocap": {"baud": 1}}, self.quad, DT, seed=0
            )
        self.assertIn("mocap", str(ctx.exception))
        self.assertIn("baud", str(ctx.exception))

    def test_bad_value_types_raise(self):
        bad = dict(NOISY_CFG)
        bad["imu"] = dict(NOISY_CFG["imu"], gyro_noise_std="loud")
        with self.assertRaises(ValueError) as ctx:
            build_estimator(bad, self.quad, DT, seed=0)
        self.assertIn("gyro_noise_std", str(ctx.exception))
        bad = dict(NOISY_CFG)
        bad["mocap"] = dict(NOISY_CFG["mocap"], rate_hz=-100.0)
        with self.assertRaises(ValueError):
            build_estimator(bad, self.quad, DT, seed=0)

    def test_lighthouse_flags_must_be_booleans(self):
        # "false" from a YAML string would otherwise be truthy and switch the flag on
        for key in ("lighthouse_bias_shared", "lighthouse_yaw_update"):
            with self.assertRaises(ValueError) as ctx:
                build_estimator({"enabled": True, key: "false"}, self.quad, DT, seed=0)
            self.assertIn(key, str(ctx.exception))
        est = build_estimator({"enabled": True, "lighthouse_bias_shared": True,
                               "lighthouse_yaw_update": True}, self.quad, DT, seed=0)
        self.assertIsInstance(est, ErrorStateNavEKF)

    def test_defaults_fill_missing_subdicts(self):
        est = build_estimator({"enabled": True}, self.quad, DT, seed=0)
        self.assertIsInstance(est, ErrorStateNavEKF)
        self.assertEqual(est.backend, "kalman")
        self.assertEqual(est.lighthouse.protocol, LH2_PROTOCOL)
        self.assertEqual(est.lighthouse.delay_steps, LH2_DELAY_STEPS)
        self.assertAlmostEqual(est.lighthouse.rate_hz, LH2_RATE_HZ)
        self.assertAlmostEqual(est.lighthouse.tilt_rad, LH2_TILT_RAD)
        self.assertAlmostEqual(est.lighthouse.angle_noise_std, LH2_ANGLE_NOISE_STD)
        self.assertAlmostEqual(est.lighthouse.angle_bias_std, LH2_ANGLE_BIAS_STD)
        complementary = build_estimator(
            {"enabled": True, "backend": "complementary"}, self.quad, DT, seed=0
        )
        self.assertIsInstance(complementary, ComplementaryEstimator)
        self.assertEqual(complementary.k_pos, 0.4)
        self.assertEqual(complementary.k_bias, 0.01)
        self.assertEqual(complementary.mocap.delay_steps, 2)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_estimator(
                {"enabled": True, "backend": "gps"}, self.quad, DT, seed=0
            )
        self.assertIn("backend", str(ctx.exception))

    def test_hardware_keys_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            build_estimator(
                {"enabled": True, "lighthouse": {"rate_hz": 100.0}},
                self.quad,
                DT,
                seed=0,
            )
        self.assertIn("lighthouse", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            build_estimator(
                {"enabled": True, "imu": {"gyro_range_dps": 2000.0}},
                self.quad,
                DT,
                seed=0,
            )
        self.assertIn("gyro_range_dps", str(ctx.exception))

    def test_gyro_range_override_is_a_top_level_key(self):
        # default: complementary uses the firmware BMI088 range, kalman the
        # research-plant range; the top-level key overrides either.
        import crazyflie_sim.model.params as params

        comp = build_estimator(
            {"enabled": True, "backend": "complementary"}, self.quad, DT, seed=0
        )
        self.assertAlmostEqual(
            comp.imu.gyro_range_rad_s, np.radians(params.gyro_range_dps)
        )
        wide = build_estimator(
            {"enabled": True, "backend": "complementary", "gyro_range_dps": 4000.0},
            self.quad,
            DT,
            seed=0,
        )
        self.assertAlmostEqual(wide.imu.gyro_range_rad_s, np.radians(4000.0))
        with self.assertRaises(ValueError):
            build_estimator(
                {"enabled": True, "backend": "complementary", "gyro_range_dps": 0.0},
                self.quad,
                DT,
                seed=0,
            )


class TestRunnerIntegration(unittest.TestCase):
    def test_estimated_run_is_seeded_reproducible_and_additive(self):
        bundle = (NOISY_CFG, 123)
        run1 = Simulation("baseline", duration_s=3.0, estimator=bundle).run()
        run2 = Simulation("baseline", duration_s=3.0, estimator=bundle).run()
        self.assertIn("est_pos", run1)
        self.assertIn("est_att", run1)
        n = run1["t"].shape[0]
        self.assertEqual(run1["est_pos"].shape, (n, 3))
        self.assertEqual(run1["est_att"].shape, (n, 3))
        np.testing.assert_array_equal(run1["est_pos"], run2["est_pos"])
        np.testing.assert_array_equal(run1["est_att"], run2["est_att"])
        np.testing.assert_array_equal(run1["pos"], run2["pos"])

    def test_disabled_estimator_leaves_no_trace(self):
        run = Simulation("baseline", duration_s=3.0, estimator=None).run()
        self.assertNotIn("est_pos", run)
        self.assertNotIn("est_att", run)


_KALMAN_ZERO_NOISE = {
    "enabled": True,
    "backend": "kalman",
    "imu": {
        "gyro_noise_std": 0.0,
        "gyro_bias_std": 0.0,
        "accel_noise_std": 0.0,
        "accel_bias_std": 0.0,
    },
}


class TestLighthouseSensor(unittest.TestCase):
    def test_rate_gating_delay_and_no_euler_rpy(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        lighthouse = Lighthouse(
            np.random.default_rng(0),
            rate_hz=100.0,
            angle_noise_std=0.0,
            angle_bias_std=0.0,
            delay_steps=2,
        )
        sample = lighthouse.update(quad, 0.0)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.angles.shape, (2, 4, 2))
        self.assertTrue(np.all(sample.valid))
        self.assertFalse(hasattr(sample, "att"))
        self.assertIsNone(lighthouse.update(quad, 0.005))
        quad.state[0:3] = [0.2, 0.0, 1.0]
        delayed = lighthouse.update(quad, 0.01)
        np.testing.assert_allclose(delayed.angles, sample.angles)
        lighthouse.update(quad, 0.02)
        later = lighthouse.update(quad, 0.03)
        self.assertFalse(np.allclose(later.angles, sample.angles))

    def test_lh1_zero_tilt_is_azimuth_and_elevation(self):
        stations = default_basestations()
        station = stations[0]
        sensor = np.array([0.35, -0.20, 0.80], dtype=float)
        azimuth = sweep_angle_and_grad(
            sensor,
            station["position"],
            station["rotation"],
            np.eye(3),
            0.0,
        )
        elevation = sweep_angle_and_grad(
            sensor,
            station["position"],
            station["rotation"],
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
            0.0,
        )
        self.assertIsNotNone(azimuth)
        self.assertIsNotNone(elevation)
        self.assertFalse(np.isclose(azimuth[0], elevation[0]))

    def test_lh2_zero_tilt_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            Lighthouse(
                np.random.default_rng(0),
                rate_hz=50.0,
                angle_noise_std=0.0,
                angle_bias_std=0.0,
                delay_steps=0,
                tilt_rad=0.0,
                protocol="lh2",
            )
        self.assertIn("tilt_rad", str(ctx.exception))

    def test_lh2_pair_recovers_azimuth(self):
        stations = default_basestations()
        station = stations[0]
        sensor = np.array([0.35, -0.20, 0.80], dtype=float)
        tilt = LH2_TILT_RAD
        plus = sweep_angle_and_grad(
            sensor, station["position"], station["rotation"], np.eye(3), tilt
        )
        minus = sweep_angle_and_grad(
            sensor, station["position"], station["rotation"], np.eye(3), -tilt
        )
        azimuth = sweep_angle_and_grad(
            sensor, station["position"], station["rotation"], np.eye(3), 0.0
        )
        self.assertIsNotNone(plus)
        self.assertIsNotNone(minus)
        self.assertIsNotNone(azimuth)
        recovered = plus[0] + 0.5 * (
            ((minus[0] - plus[0] + np.pi) % (2.0 * np.pi)) - np.pi
        )
        self.assertAlmostEqual(recovered, azimuth[0], places=9)
        self.assertFalse(np.isclose(plus[0], minus[0]))

    def test_triangulation_recovers_com_from_one_photodiode(self):
        from crazyflie_sim.model.lighthouse import triangulate_position

        quad = Quadcopter((0.15, -0.25, 1.05), (0.04, -0.03, 0.5))
        rot = RPYToRot(*[float(a) for a in quad.attitude()]).T
        for protocol, tilt in ((LH2_PROTOCOL, LH2_TILT_RAD), ("lh1", 0.0)):
            lighthouse = Lighthouse(
                np.random.default_rng(0),
                rate_hz=LH2_RATE_HZ,
                angle_noise_std=0.0,
                angle_bias_std=0.0,
                delay_steps=0,
                tilt_rad=tilt,
                protocol=protocol,
            )
            angles, valid = lighthouse.predict(quad.position(), rot)
            recovered = triangulate_position(
                angles,
                valid,
                lighthouse.basestations,
                lighthouse.sweeps,
                lighthouse.sensors_body,
                rot,
                0,
            )
            np.testing.assert_allclose(
                recovered, quad.position(), atol=1e-9, err_msg=protocol
            )


class TestErrorStateNavEKF(unittest.TestCase):
    def test_healthy_hover_tracks_truth(self):
        run = Simulation("baseline", duration_s=2.0, estimator=(_KALMAN_ZERO_NOISE, 42)).run()
        mask = run["t"] >= 1.0
        pos_err = np.linalg.norm(run["est_pos"][mask] - run["pos"][mask], axis=1)
        att_err = np.linalg.norm(run["est_att"][mask] - run["att"][mask], axis=1)
        self.assertLess(float(pos_err.max()), 0.12)
        self.assertLess(float(att_err.max()), 5e-2)
        self.assertLess(float(run["err"][-1]), 0.10)

    def test_delayed_lighthouse_yaw_spin_does_not_walk_bias_or_heading(self):
        dt = DT
        spin = 40.0
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        estimator = build_estimator(_KALMAN_ZERO_NOISE, quad, dt, seed=0)
        self.assertGreater(estimator.imu.gyro_range_rad_s, spin)
        for step in range(int(3.0 / dt)):
            phi, theta, psi = quad.attitude()
            eulerdot = Quadcopter.euler_rates(phi, theta, 0.0, 0.0, spin)
            quad.state[6:9] = quad.state[6:9] + dt * eulerdot
            quad.state[9:12] = np.array([0.0, 0.0, spin], dtype=float)
            estimator.update(
                quad, dt, float(step + 1) * dt, np.zeros(3, dtype=float)
            )
        self.assertLess(abs(float(estimator.gyro_bias_est[2])), 0.20)
        self.assertLess(_tilt_deg(estimator.att_est), 5.0)
        yaw_err = abs(
            float(_wrap_heading(estimator.att_est[2] - quad.attitude()[2]))
        )
        self.assertLess(yaw_err, 0.25)

    def test_spinning_cone_tracks_tilt_without_walking_yaw_bias(self):
        dt = DT
        spin = 40.0
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        estimator = build_estimator(_KALMAN_ZERO_NOISE, quad, dt, seed=0)
        for step in range(int(1.0 / dt)):
            phi, theta, psi = quad.attitude()
            eulerdot = Quadcopter.euler_rates(phi, theta, 0.0, 0.0, spin)
            quad.state[6:9] = quad.state[6:9] + dt * eulerdot
            quad.state[9:12] = np.array([0.0, 0.0, spin], dtype=float)
            estimator.update(
                quad, dt, float(step + 1) * dt, np.zeros(3, dtype=float)
            )
        cone = np.radians(8.0)
        for step in range(int(1.0 / dt), int(2.5 / dt)):
            t = float(step + 1) * dt
            quad.state[6] = cone
            quad.state[7] = 0.0
            quad.state[8] = spin * t
            quad.state[9:12] = np.array(
                [0.0, spin * np.sin(cone), spin * np.cos(cone)], dtype=float
            )
            estimator.update(quad, dt, t, np.zeros(3, dtype=float))
        self.assertLess(abs(float(estimator.gyro_bias_est[2])), 0.20)
        self.assertLess(abs(_tilt_deg(estimator.att_est) - np.degrees(cone)), 8.0)
        yaw_err = abs(
            float(_wrap_heading(estimator.att_est[2] - quad.attitude()[2]))
        )
        self.assertLess(yaw_err, 0.35)

    def test_research_gyro_range_does_not_saturate_forty_rad_s(self):
        quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        estimator = build_estimator(_KALMAN_ZERO_NOISE, quad, DT, seed=0)
        measured = estimator.imu.gyro(np.array([0.0, 0.0, 40.0]))
        self.assertFalse(bool(estimator.imu.last_gyro_saturated[2]))
        self.assertAlmostEqual(float(measured[2]), 40.0)


class TestLighthouseKalmanAccuracyClass(unittest.TestCase):
    """Taffanel 2021 centimetre-class pose error, not millimetre tracking.

    The band matches that published accuracy class, not a pointwise
    reproduction of the Taffanel dataset.
    """

def _tilt_deg(att):
    att = np.asarray(att, dtype=float).reshape(3,)
    return float(
        np.degrees(
            np.arccos(np.clip(np.cos(att[0]) * np.cos(att[1]), -1.0, 1.0))
        )
    )


def _wrap_heading(angle):
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


if __name__ == "__main__":
    unittest.main()
