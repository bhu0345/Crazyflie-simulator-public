import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import crazyflie_sim.model.params as params
from crazyflie_sim.experiments import monte_carlo as mc
from crazyflie_sim.model.actuator_profile import CRAZYFLIE21_PHYSICAL as CF, get_actuator_profile
from crazyflie_sim.sim.config import PARAMS_OVERRIDE_KEYS, RunConfig
from crazyflie_sim.sim.scenario import Scenario

VALID_SPEC = {
    "n_runs": 3,
    "seed": 123,
    "perturb": {
        "mass": {"dist": "uniform", "low": 0.038, "high": 0.042},
        "aero_drag_factor_xy": {
            "dist": "normal",
            "mean": 0.00539,
            "std": 0.0005,
            "clip": [0.004, 0.007],
        },
        "I": {"dist": "uniform", "low": 0.9, "high": 1.1, "mode": "scale"},
    },
}


def _spec(**changes):
    spec = copy.deepcopy(VALID_SPEC)
    spec.update(changes)
    return spec


def _perturb(**changes):
    spec = copy.deepcopy(VALID_SPEC)
    spec["perturb"].update(changes)
    return spec


def _strip_wall_time(obj):
    """Recursively drop every "wall_time_s" dict entry (machine-dependent)."""
    if isinstance(obj, dict):
        return {
            k: _strip_wall_time(v) for k, v in obj.items() if k != "wall_time_s"
        }
    if isinstance(obj, list):
        return [_strip_wall_time(v) for v in obj]
    return obj


def _canonical(obj):
    """_strip_wall_time + map non-finite floats to strings so == works
    (float('nan') != float('nan') would make identical reports compare
    unequal; the written JSON files already map non-finite to None)."""
    obj = _strip_wall_time(obj)
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return repr(obj)
    return obj


class TestValidateMonteCarloSpec(unittest.TestCase):
    def test_valid_spec_normalizes(self):
        out = mc.validate_monte_carlo_spec(VALID_SPEC)
        self.assertEqual(out["n_runs"], 3)
        self.assertEqual(out["seed"], 123)
        self.assertEqual(out["perturb"]["mass"]["mode"], "absolute")  # default
        self.assertEqual(out["perturb"]["mass"]["dist"], "uniform")
        self.assertEqual(out["perturb"]["I"]["mode"], "scale")
        self.assertIsInstance(out["perturb"]["mass"]["low"], float)

    def test_n_runs_too_small(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(_spec(n_runs=1))

    def test_n_runs_not_int(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(_spec(n_runs=2.5))

    def test_seed_missing(self):
        spec = _spec()
        del spec["seed"]
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(spec)

    def test_seed_not_int(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(_spec(seed="123"))

    def test_unknown_top_level_key(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(_spec(replicates=3))

    def test_bad_dist(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(
                _perturb(mass={"dist": "gaussian", "low": 0.038, "high": 0.042})
            )

    def test_uniform_missing_high(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(_perturb(mass={"dist": "uniform", "low": 0.038}))

    def test_normal_missing_std(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(
                _perturb(mass={"dist": "normal", "mean": 0.04})
            )

    def test_normal_negative_std(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(
                _perturb(mass={"dist": "normal", "mean": 0.04, "std": -0.001})
            )

    def test_uniform_low_above_high(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(
                _perturb(mass={"dist": "uniform", "low": 0.042, "high": 0.038})
            )

    def test_bad_clip(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(
                _perturb(
                    mass={"dist": "normal", "mean": 0.04, "std": 0.001, "clip": [0.05, 0.03]}
                )
            )

    def test_non_whitelist_key(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(
                _perturb(drag_area_xy={"dist": "uniform", "low": 0.006, "high": 0.01})
            )

    def test_inertia_absolute_mode_rejected(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(
                _perturb(I={"dist": "uniform", "low": 2.0e-5, "high": 3.0e-5, "mode": "absolute"})
            )

    def test_inertia_default_mode_rejected(self):
        # "I" without an explicit mode defaults to "absolute" -> rejected.
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(
                _perturb(I={"dist": "uniform", "low": 0.9, "high": 1.1})
            )

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(
                _perturb(mass={"dist": "uniform", "low": 0.038, "high": 0.042, "mode": "relative"})
            )

    def test_unknown_field_for_dist(self):
        with self.assertRaises(ValueError):
            mc.validate_monte_carlo_spec(
                _perturb(mass={"dist": "uniform", "low": 0.038, "high": 0.042, "mean": 0.04})
            )


class TestSamplePerturbations(unittest.TestCase):
    def test_same_seed_identical_sequences(self):
        spec = mc.validate_monte_carlo_spec(VALID_SPEC)
        rng_a = np.random.default_rng(spec["seed"])
        rng_b = np.random.default_rng(spec["seed"])
        seq_a = [mc.sample_perturbations(spec["perturb"], rng_a) for _ in range(5)]
        seq_b = [mc.sample_perturbations(spec["perturb"], rng_b) for _ in range(5)]
        self.assertEqual(seq_a, seq_b)

    def test_different_seed_differs(self):
        spec = mc.validate_monte_carlo_spec(VALID_SPEC)
        rng_a = np.random.default_rng(spec["seed"])
        rng_c = np.random.default_rng(spec["seed"] + 1)
        seq_a = [mc.sample_perturbations(spec["perturb"], rng_a) for _ in range(5)]
        seq_c = [mc.sample_perturbations(spec["perturb"], rng_c) for _ in range(5)]
        self.assertNotEqual(seq_a, seq_c)

    def test_samples_respect_bounds_and_shapes(self):
        spec = mc.validate_monte_carlo_spec(VALID_SPEC)
        rng = np.random.default_rng(999)
        nominal_diag = np.diag(np.asarray(CF.inertia, dtype=float))
        for _ in range(10):
            overrides = mc.sample_perturbations(spec["perturb"], rng)
            self.assertTrue(0.038 <= overrides["mass"] <= 0.042)
            self.assertTrue(0.004 <= overrides["aero_drag_factor_xy"] <= 0.007)  # clip
            self.assertEqual(len(overrides["I"]), 3)
            for value, nominal in zip(overrides["I"], nominal_diag):
                ratio = value / nominal
                self.assertTrue(0.9 <= ratio <= 1.1)

    def test_scalar_scale_mode_multiplies_nominal(self):
        spec = mc.validate_monte_carlo_spec(
            _spec(
                perturb={
                    "motor_tau_up": {
                        "dist": "uniform",
                        "low": 0.9,
                        "high": 1.1,
                        "mode": "scale",
                    }
                }
            )
        )
        rng = np.random.default_rng(5)
        for _ in range(5):
            overrides = mc.sample_perturbations(spec["perturb"], rng)
            ratio = overrides["motor_tau_up"] / CF.tau_up
            self.assertTrue(0.9 <= ratio <= 1.1)

    def test_scale_mode_uses_the_run_profile(self):
        spec = mc.validate_monte_carlo_spec(_spec(perturb={
            "mass": {"dist": "uniform", "low": 1.0, "high": 1.0, "mode": "scale"},
            "I": {"dist": "uniform", "low": 1.0, "high": 1.0, "mode": "scale"},
        }))
        standard = get_actuator_profile("crazyflie21_standard")
        out = mc.sample_perturbations(spec["perturb"], np.random.default_rng(0), "crazyflie21_standard")
        self.assertEqual(out["mass"], standard.mass)
        self.assertEqual(out["I"], list(np.diag(standard.inertia)))

    def test_vehicle_keys_go_to_the_plant(self):
        plant, rest = mc.split_overrides({"mass": 0.041, "I": [1e-5, 2e-5, 3e-5],
                                          "motor_tau_up": 0.06, "aero_drag_factor_xy": 0.005})
        self.assertEqual(plant, {"mass": 0.041, "inertia": [1e-5, 2e-5, 3e-5], "motor_tau_up": 0.06})
        self.assertEqual(rest, {"aero_drag_factor_xy": 0.005})


class TestPerturbedParams(unittest.TestCase):
    def _snapshot(self):
        snap = {}
        for key in PARAMS_OVERRIDE_KEYS:
            if not hasattr(params, key):
                continue
            value = getattr(params, key)
            snap[key] = value.copy() if isinstance(value, np.ndarray) else value
        return snap

    def test_apply_and_bit_exact_restore(self):
        saved = self._snapshot()
        overrides = {
            "aero_drag_factor_xy": 0.0061,
            "battery_voltage_nominal": 3.9,
        }
        with mc.perturbed_params(overrides):
            self.assertEqual(params.aero_drag_factor_xy, 0.0061)
            self.assertEqual(params.battery_voltage_nominal, 3.9)
        # Bit-exact restore of every whitelisted attribute.
        for key, value in saved.items():
            current = getattr(params, key)
            if isinstance(value, np.ndarray):
                self.assertTrue(np.array_equal(current, value), key)
            else:
                self.assertTrue(current == value, key)

    def test_restore_on_exception(self):
        saved = self._snapshot()
        with self.assertRaises(RuntimeError):
            with mc.perturbed_params({"battery_sag_tau": 0.099}):
                self.assertEqual(params.battery_sag_tau, 0.099)
                raise RuntimeError("boom")
        self.assertTrue(params.battery_sag_tau == saved["battery_sag_tau"])


class TestAggregateMetrics(unittest.TestCase):
    def test_mean_std_n(self):
        reps = [
            {"m1": {"a": 1.0, "b": 2.0, "c": float("nan")}, "m2": {"a": 10.0}},
            {"m1": {"a": 2.0, "b": float("nan"), "c": float("nan")}, "m2": {"a": 20.0}},
            {"m1": {"a": float("nan"), "b": 6.0, "c": float("nan")}, "m2": {"a": float("nan")}},
        ]
        agg = mc.aggregate_metrics(reps)
        self.assertEqual(set(agg), {"m1", "m2"})

        a = agg["m1"]["a"]
        self.assertEqual(a["n"], 2)
        self.assertAlmostEqual(a["mean"], 1.5)
        self.assertAlmostEqual(a["std"], 0.5)
        self.assertEqual(len(a["values"]), 3)

        b = agg["m1"]["b"]
        self.assertEqual(b["n"], 2)
        self.assertAlmostEqual(b["mean"], 4.0)
        self.assertAlmostEqual(b["std"], 2.0)

        m2a = agg["m2"]["a"]
        self.assertEqual(m2a["n"], 2)
        self.assertAlmostEqual(m2a["mean"], 15.0)
        self.assertAlmostEqual(m2a["std"], 5.0)

        # All-NaN metric: n = 0, mean/std None (no RuntimeWarning from nanmean).
        c = agg["m1"]["c"]
        self.assertEqual(c["n"], 0)
        self.assertIsNone(c["mean"])
        self.assertIsNone(c["std"])


class TestRunMonteCarloEndToEnd(unittest.TestCase):
    """Mini Monte Carlo through experiments.compare.run_monte_carlo."""

    def test_report_structure_and_reproducibility(self):
        import contextlib
        import io

        from crazyflie_sim.experiments.compare import run_monte_carlo

        cfg = RunConfig(
            name="mc",
            scenario=Scenario(sim_mode="hover_wind", duration_s=2.0),
            monte_carlo={
                "n_runs": 3,
                "seed": 11,
                "perturb": {
                    "mass": {"dist": "uniform", "low": 0.038, "high": 0.042},
                    "I": {"dist": "uniform", "low": 0.95, "high": 1.05, "mode": "scale"},
                    "motor_tau_up": {"dist": "normal", "mean": 0.05, "std": 0.004, "clip": [0.04, 0.06]},
                },
            },
        )
        modes = ["baseline", "mellinger"]
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / "run1", Path(tmp) / "run2"]
            reports = []
            for d in dirs:
                d.mkdir()
                # The second report comes from two worker processes: it must
                # match the sequential one exactly.
                with contextlib.redirect_stdout(io.StringIO()):
                    reports.append(run_monte_carlo(cfg, modes, d, figures=(d is dirs[0]),
                                                   jobs=1 if d is dirs[0] else 2))
            r1, r2 = reports
            self.assertEqual(set(r1), {"spec", "replicates", "aggregate"})
            self.assertEqual(len(r1["replicates"]), 3)
            for i, rep in enumerate(r1["replicates"]):
                self.assertEqual(rep["index"], i)
                self.assertEqual(len(rep["plant_overrides"]["inertia"]), 3)
                self.assertEqual(rep["params_overrides"], {})
                self.assertEqual(set(rep["metrics"]), set(modes))
            agg = r1["aggregate"]["baseline"]["rms_err_post_wind_m"]
            self.assertEqual(agg["n"], 3)
            self.assertEqual(_canonical(r1), _canonical(r2))
            j1 = json.loads((dirs[0] / "monte_carlo_report.json").read_text())
            j2 = json.loads((dirs[1] / "monte_carlo_report.json").read_text())
            self.assertEqual(_strip_wall_time(j1), _strip_wall_time(j2))
            for mode in modes:
                self.assertTrue((dirs[0] / "replicate_0" / "{}.npz".format(mode)).exists())


if __name__ == "__main__":
    unittest.main()
