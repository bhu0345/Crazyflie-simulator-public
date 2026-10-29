import unittest

import numpy as np

from crazyflie_sim.model.wind import AxisWind, WindModel

OFF = AxisWind(mode="off")


def wind(x=OFF, y=OFF, z=OFF, **kwargs):
    return WindModel(x=x, y=y, z=z, **kwargs)


class TestAnalyticModes(unittest.TestCase):
    def test_off_is_zero_even_with_parameters(self):
        w = wind(x=AxisWind(mode="off", amplitude=0.4, constant=0.2, impulse_magnitude=0.8))
        for t in (2.0, 2.5, 10.0):
            np.testing.assert_array_equal(w.value(t), np.zeros(3))

    def test_sin_matches_formula_exactly(self):
        axis = AxisWind(mode="sin", amplitude=0.4, frequency_hz=0.25, phase=np.pi / 2, offset=0.05)
        w = wind(x=axis)
        for t in (2.0, 2.3, 3.7, 9.1):
            expected = 0.05 + 0.4 * np.sin(2.0 * np.pi * 0.25 * (t - 2.0) + np.pi / 2)
            self.assertEqual(w.value(t)[0], expected)

    def test_constant_square_impulse(self):
        w = wind(
            x=AxisWind(mode="constant", constant=0.2),
            y=AxisWind(mode="square", square_high=0.3, square_low=-0.3, square_duty=0.5, frequency_hz=0.5),
            z=AxisWind(mode="impulse", impulse_magnitude=0.2, impulse_time=1.0, impulse_width=0.05),
        )
        self.assertEqual(w.value(2.5).tolist(), [0.2, 0.3, 0.0])
        self.assertEqual(w.value(3.0).tolist(), [0.2, -0.3, 0.2])
        self.assertEqual(w.value(3.2)[1], -0.3)
        self.assertEqual(w.value(3.06)[2], 0.0)
        self.assertEqual(w.value(4.2)[1], 0.3)
        self.assertEqual(w.value(3.5)[1], -0.3)

    def test_smoothstep_ramp(self):
        w = wind(x=AxisWind(mode="smoothstep", smoothstep_magnitude=0.3, smoothstep_rise_time=0.5))
        self.assertEqual(w.value(2.0)[0], 0.0)
        self.assertEqual(w.value(2.25)[0], 0.15)
        self.assertEqual(w.value(2.5)[0], 0.3)
        self.assertEqual(w.value(9.0)[0], 0.3)

    def test_time_window(self):
        w = wind(x=AxisWind(mode="constant", constant=0.2), start_time=1.0, end_time=2.0)
        self.assertEqual(w.value(0.999)[0], 0.0)
        self.assertEqual(w.value(1.5)[0], 0.2)
        self.assertEqual(w.value(2.001)[0], 0.0)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            AxisWind(mode="gusty")


class TestDryden(unittest.TestCase):
    AXIS = AxisWind(mode="dryden", dryden_sigma=0.15, dryden_tau=1.0)
    TIMES = [2.0 + 0.002 * k for k in range(200)]

    def test_same_seed_reproduces_and_seeds_differ(self):
        def sequence(seed):
            w = wind(x=self.AXIS, dryden_seed=seed)
            return [w.value(t)[0] for t in self.TIMES]

        self.assertEqual(sequence(0), sequence(0))
        self.assertNotEqual(sequence(0), sequence(1))

    def test_repeated_time_returns_cached_value(self):
        w = wind(x=self.AXIS)
        first = w.value(2.5).copy()
        np.testing.assert_array_equal(w.value(2.5), first)
        later = w.value(2.6)
        w.reset()
        w.value(2.5)
        np.testing.assert_array_equal(w.value(2.6), later)

    def test_backwards_time_restarts_the_sequence(self):
        w = wind(x=self.AXIS)
        first = [w.value(t)[0] for t in self.TIMES]
        second = [w.value(t)[0] for t in self.TIMES]
        self.assertEqual(first, second)

    def test_gated_calls_do_not_consume_draws(self):
        w = wind(x=self.AXIS)
        for t in (0.0, 1.0, 1.999):
            np.testing.assert_array_equal(w.value(t), np.zeros(3))
        gated = [w.value(t)[0] for t in self.TIMES]
        w.reset()
        self.assertEqual(gated, [w.value(t)[0] for t in self.TIMES])

    def test_other_axes_stay_exact(self):
        w = wind(x=self.AXIS, y=AxisWind(mode="sin", amplitude=0.3, frequency_hz=0.2, phase=0.1, offset=0.05))
        for t in self.TIMES[::20]:
            expected = 0.05 + 0.3 * np.sin(2.0 * np.pi * 0.2 * (t - 2.0) + 0.1)
            self.assertEqual(w.value(t)[1], expected)


class TestForceVersusVelocity(unittest.TestCase):
    def test_channel_switch(self):
        w = wind(x=AxisWind(mode="constant", constant=0.3))
        np.testing.assert_array_equal(w.force_world(3.0), [0.3, 0.0, 0.0])
        self.assertIsNone(w.velocity_world(3.0))
        w.as_velocity = True
        np.testing.assert_array_equal(w.force_world(3.0), np.zeros(3))
        np.testing.assert_array_equal(w.velocity_world(3.0), [0.3, 0.0, 0.0])


class TestConfig(unittest.TestCase):
    def test_defaults_are_three_sinusoids(self):
        w = WindModel()
        self.assertEqual((w.x.mode, w.y.mode, w.z.mode), ("sin", "sin", "sin"))
        self.assertEqual((w.x.amplitude, w.x.frequency_hz), (0.4, 0.1))
        self.assertEqual(w.start_time, 2.0)

    def test_from_dict_overrides_one_axis(self):
        w = WindModel.from_dict({"start_time": 1.35, "x": {"mode": "smoothstep", "smoothstep_magnitude": 0.3}})
        self.assertEqual(w.start_time, 1.35)
        self.assertEqual(w.x.mode, "smoothstep")
        self.assertEqual(w.y, WindModel().y)

    def test_uppercase_keys_point_to_the_new_schema(self):
        with self.assertRaisesRegex(ValueError, "UPPERCASE"):
            WindModel.from_dict({"FX_MODE": "sin"})
        with self.assertRaisesRegex(ValueError, "wind.x"):
            WindModel.from_dict({"x": {"amplitud": 1.0}})

    def test_formula_text(self):
        text = wind(x=AxisWind(mode="dryden", dryden_sigma=0.1), y=AxisWind(mode="smoothstep")).formula_text()
        self.assertIn("OU(", text)
        self.assertIn("smoothstep", text)
        self.assertIn("VELOCITY", wind(as_velocity=True).formula_text())


if __name__ == "__main__":
    unittest.main()
