"""The standard benchmark suite (experiments/benchmark.py, configs/benchmark/)."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from crazyflie_sim.controllers import registry
from crazyflie_sim.controllers.base import ControllerBase
from crazyflie_sim.experiments import benchmark
from crazyflie_sim.sim.config import load_config
from crazyflie_sim.sim.scenario import Scenario

SUITE = ("hover_gust", "trajectory", "partial_loss", "complete_loss",
         "estimator_gust", "model_mismatch", "control_link")


def _run(t, err, z=1.0, tilt_rad=0.0):
    n = len(t)
    pos = np.zeros((n, 3))
    pos[:, 2] = z
    att = np.zeros((n, 3))
    att[:, 0] = tilt_rad
    return {"t": t, "pos": pos, "err": np.asarray(err, dtype=float), "att": att,
            "vel": np.zeros((n, 3)), "omega": np.zeros((n, 3)),
            "motor_pwm": np.full((n, 4), 30000.0)}


class TestSuite(unittest.TestCase):
    def test_the_suite_is_the_numbered_configs(self):
        cases = benchmark.suite()
        self.assertEqual(tuple(cases), SUITE)
        self.assertTrue(all(p.name[0].isdigit() for p in cases.values()))
        self.assertEqual(tuple(benchmark.suite(["partial_loss", "hover_gust"])),
                         ("hover_gust", "partial_loss"))
        with self.assertRaisesRegex(ValueError, "unknown benchmark scenario"):
            benchmark.suite(["nope"])


class TestSeeds(unittest.TestCase):
    def test_only_scenarios_with_noise_turbulence_or_a_lossy_link_are_replicated(self):
        stochastic = {name for name, path in benchmark.suite().items()
                      if benchmark.stochastic(load_config(path).scenario)}
        self.assertEqual(stochastic, {"hover_gust", "estimator_gust", "model_mismatch",
                                      "control_link"})

    def test_replicate_k_adds_k_to_every_seed(self):
        scenario = load_config(benchmark.suite()["control_link"]).scenario
        self.assertIs(benchmark.replicate(scenario, 0), scenario)
        third = benchmark.replicate(scenario, 2)
        self.assertEqual(third.seed, scenario.seed + 2)
        self.assertEqual(third.wind.dryden_seed, scenario.wind.dryden_seed + 2)
        self.assertEqual(third.control_link["seed"], scenario.control_link["seed"] + 2)
        self.assertEqual(third.control_link["dropout_probability"],
                         scenario.control_link["dropout_probability"])
        self.assertEqual(scenario.wind.dryden_seed, 7)           # the original is untouched


def _seeds(rms, survived=True):
    return [{"rms_err_m": r, "survived": survived} for r in rms]


class TestStatistics(unittest.TestCase):
    def test_paired_ratio_is_the_geometric_mean_over_the_seeds(self):
        ref = _seeds([0.10, 0.20, 0.40, 0.10, 0.30])
        s = benchmark.summary(_seeds([0.05, 0.10, 0.20, 0.05, 0.15]), ref)
        self.assertAlmostEqual(s["ratio"], 0.5)
        self.assertAlmostEqual(s["ratio_ci"][0], 0.5)              # every seed halves the error
        self.assertEqual(benchmark.verdict(s), "better")
        self.assertAlmostEqual(s["rms_median_m"], 0.10)
        self.assertLessEqual(s["rms_ci_m"][0], 0.10)
        self.assertGreaterEqual(s["rms_ci_m"][1], 0.10)
        self.assertEqual(benchmark.verdict(benchmark.summary(ref, ref)), "same")

    def test_an_inconsistent_difference_is_not_significant(self):
        ref = _seeds([0.10] * 6)
        s = benchmark.summary(_seeds([0.05, 0.20, 0.06, 0.18, 0.07, 0.15]), ref)
        self.assertLess(s["ratio_ci"][0], 1.0)
        self.assertGreater(s["ratio_ci"][1], 1.0)
        self.assertEqual(benchmark.verdict(s), "same")

    def test_a_single_deterministic_run_is_compared_exactly(self):
        s = benchmark.summary(_seeds([0.12]), _seeds([0.10]))
        self.assertEqual(s["ratio_ci"], [s["ratio"], s["ratio"]])
        self.assertEqual(benchmark.verdict(s), "worse")

    def test_errors_are_left_out_of_the_statistics_and_the_score(self):
        failed = {"rms_err_m": float("nan"), "survived": False, "error": "RuntimeError: x"}
        s = benchmark.summary(_seeds([0.1, 0.1]) + [failed], _seeds([0.1, 0.1, 0.1]))
        self.assertEqual((s["seeds"], s["survived"], s["errors"]), (3, 2, 1))
        self.assertAlmostEqual(s["ratio"], 1.0)
        everything_failed = benchmark.summary([failed, failed], _seeds([0.1, 0.1]))
        self.assertTrue(np.isnan(everything_failed["ratio"]))
        score = benchmark.scores({"a": {"m": s}, "b": {"m": everything_failed}})
        self.assertTrue(np.isnan(score["m"]))                     # no score without every scenario


class TestScore(unittest.TestCase):
    scenario = Scenario(sim_mode="hover_wind")  # wind (the event) from 2 s

    def test_metrics_start_at_the_event_and_settle(self):
        t = np.arange(0.0, 10.0, 0.002)
        err = np.where(t < 2.0, 5.0, np.where(t < 4.0, 0.3, 0.02))  # pre-event error ignored
        s = benchmark.score(self.scenario, _run(t, err))
        self.assertTrue(s["survived"])
        self.assertAlmostEqual(s["max_err_m"], 0.3)
        self.assertAlmostEqual(s["final_err_m"], 0.02)
        self.assertAlmostEqual(s["settle_s"], 2.0, places=2)
        self.assertAlmostEqual(s["saturation_pct"], 0.0)

    def test_ground_contact_and_tumbling_are_crashes(self):
        t = np.arange(0.0, 4.0, 0.002)
        self.assertFalse(benchmark.score(self.scenario, _run(t, np.zeros_like(t), z=0.0))["survived"])
        self.assertFalse(benchmark.score(
            self.scenario, _run(t, np.zeros_like(t), tilt_rad=np.radians(120.0)))["survived"])
        never = benchmark.score(self.scenario, _run(t, np.full_like(t, 0.5)))
        self.assertTrue(never["survived"])
        self.assertTrue(np.isnan(never["settle_s"]))

    def test_saturation_counts_the_scoring_window_only(self):
        t = np.arange(0.0, 8.0, 0.002)
        run = _run(t, np.zeros_like(t))
        run["motor_pwm"][t >= 2.0] = 65535.0           # saturated from the event on
        self.assertAlmostEqual(benchmark.score(self.scenario, run)["saturation_pct"], 100.0)


class _Raises(ControllerBase):
    def __init__(self, dt):
        self.dt = dt

    def set_dt(self, dt):
        self.dt = dt

    def reset(self):
        pass

    def diagnostics(self):
        return {}

    def run(self, quad, des_state, sim_time=0.0):
        if sim_time > 0.5:
            raise RuntimeError("controller bug")
        return np.full(4, 30000.0)


class TestBenchmarkRun(unittest.TestCase):
    def test_short_run_writes_the_scorecard(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = benchmark.benchmark(
                ["mellinger", "baseline"], ["hover_gust", "partial_loss"], jobs=1,
                duration_s=2.5, out_dir=tmp, progress=False, seeds=2)
            self.assertEqual(list(results), ["hover_gust", "partial_loss"])
            self.assertEqual(list(results["hover_gust"]), ["mellinger", "baseline"])
            self.assertEqual(len(results["hover_gust"]["baseline"]), 2)  # Dryden: two seeds
            self.assertEqual(len(results["partial_loss"]["baseline"]), 1)  # deterministic: one
            gust = [m["rms_err_m"] for m in results["hover_gust"]["baseline"]]
            self.assertNotEqual(gust[0], gust[1])
            for row in results.values():
                for per_seed in row.values():
                    for metrics in per_seed:
                        self.assertEqual(tuple(metrics), benchmark.METRICS + ("error",))
                        self.assertTrue(metrics["survived"])
                        self.assertIsNone(metrics["error"])
            out = Path(tmp)
            for name in ("scorecard.md", "scorecard.json", "scorecard.png",
                         "hover_gust/overview.png", "partial_loss/baseline.npz"):
                self.assertTrue((out / name).exists(), name)
            text = (out / "scorecard.md").read_text()
            self.assertIn("| scenario | seeds | Mellinger | Cascaded PID |", text)
            self.assertIn("| survived | | 2/2 | 2/2 |", text)
            self.assertIn("| score vs Cascaded PID | | ", text)
            self.assertIn("### hover_gust (2 seeds)", text)

    def test_the_reference_is_flown_when_left_out(self):
        results = benchmark.benchmark(["mellinger"], ["partial_loss"], jobs=1, duration_s=2.5,
                                      progress=False)
        self.assertEqual(list(results["partial_loss"]), ["mellinger", "baseline"])

    def test_a_controller_that_raises_fails_its_runs_not_the_suite(self):
        registry.register("_raises", label="Raises", color="#000000")(_Raises)
        self.addCleanup(registry._REGISTRY.pop, "_raises")
        with tempfile.TemporaryDirectory() as tmp, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            results = benchmark.benchmark(["_raises", "baseline"], ["partial_loss"], jobs=1,
                                          duration_s=2.5, out_dir=tmp, progress=False)
            text = (Path(tmp) / "scorecard.md").read_text()
        failed = results["partial_loss"]["_raises"][0]
        self.assertFalse(failed["survived"])
        self.assertEqual(failed["error"], "RuntimeError: controller bug")
        self.assertTrue(np.isnan(failed["rms_err_m"]))
        self.assertTrue(results["partial_loss"]["baseline"][0]["survived"])
        self.assertIn("controller bug", stderr.getvalue())         # the traceback is printed
        self.assertIn("| partial_loss | 1 | error |", text)
        self.assertIn("- Raises seed 0: `RuntimeError: controller bug`", text)


if __name__ == "__main__":
    unittest.main()
