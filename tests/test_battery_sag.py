import unittest

import numpy as np

import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.params as params
from crazyflie_sim.model.quadcopter import Quadcopter
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF


PARAMS_ATTRS = (
    "battery_sag_enabled",
    "battery_internal_resistance",
    "battery_max_current_draw",
    "battery_sag_tau",
    "battery_voltage_nominal",
    "battery_voltage_min",
    "battery_voltage_max",
)


class TestBatterySag(unittest.TestCase):
    def setUp(self):
        self._saved = {name: getattr(params, name) for name in PARAMS_ATTRS}
        params.battery_sag_enabled = False
        params.battery_internal_resistance = 0.08
        params.battery_max_current_draw = 12.0
        params.battery_sag_tau = 0.5

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(params, name, value)

    @staticmethod
    def _hover_pwm(voltage):
        return actuators.wrench_to_pwm(
            CF.mass * params.g,
            np.zeros((3, 1), dtype=float),
            voltage,
        )

    def test_sag_disabled_keeps_voltage_exactly_constant(self):
        quad = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        hover_pwm = self._hover_pwm(quad.battery_voltage)
        initial_voltage = quad.battery_voltage
        for _ in range(50):
            quad.update(0.01, hover_pwm, external_force_world=np.zeros(3))
        self.assertEqual(quad.battery_voltage, initial_voltage)
        self.assertEqual(quad.battery_voltage, params.battery_voltage_nominal)

    def test_ctor_kwarg_overrides_params_default(self):
        params.battery_sag_enabled = True
        quad = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), battery_sag=False)
        hover_pwm = self._hover_pwm(quad.battery_voltage)
        for _ in range(50):
            quad.update(0.01, hover_pwm, external_force_world=np.zeros(3))
        self.assertEqual(quad.battery_voltage, params.battery_voltage_nominal)

    def test_sag_drops_under_high_load_and_recovers_toward_ocv(self):
        quad = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), battery_sag=True)
        ocv = quad.battery_voltage
        self.assertEqual(ocv, params.battery_voltage_nominal)

        full_pwm = np.full(4, params.pwm_max, dtype=float)
        for _ in range(200):  # 2 s at full throttle
            quad.update(0.01, full_pwm, external_force_world=np.zeros(3))
        v_sagged = quad.battery_voltage
        self.assertLess(v_sagged, ocv)

        hover_pwm = self._hover_pwm(params.battery_voltage_nominal)
        for _ in range(200):  # 2 s back at hover load
            quad.update(0.01, hover_pwm, external_force_world=np.zeros(3))
        v_recovered = quad.battery_voltage
        self.assertGreater(v_recovered, v_sagged + 0.05)
        self.assertLess(v_recovered, ocv)

    def test_v_target_clamped_at_battery_voltage_min(self):
        params.battery_internal_resistance = 1000.0  # extreme: OCV - I*R << min
        params.battery_sag_tau = 0.01  # == dt: lag converges in one step
        quad = Quadcopter((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), battery_sag=True)
        hover_pwm = self._hover_pwm(quad.battery_voltage)
        quad.update(0.01, hover_pwm, external_force_world=np.zeros(3))
        self.assertAlmostEqual(
            quad.battery_voltage, params.battery_voltage_min, places=9
        )

    def test_ctor_voltage_kwarg_is_the_ocv_when_sagging(self):
        quad = Quadcopter(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            battery_voltage=4.0,
            battery_sag=True,
        )
        self.assertEqual(quad._battery_ocv, 4.0)
        hover_pwm = self._hover_pwm(4.0)
        for _ in range(200):
            quad.update(0.01, hover_pwm, external_force_world=np.zeros(3))
        # Sags below the 4.0 V OCV but above the nominal-OCV hover target.
        self.assertLess(quad.battery_voltage, 4.0)
        self.assertGreater(quad.battery_voltage, params.battery_voltage_nominal - 0.5)


if __name__ == "__main__":
    unittest.main()
