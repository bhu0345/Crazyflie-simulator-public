"""Check the installed distribution, published example, and recorded playback."""

import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]


class TestPublicDocumentation(unittest.TestCase):
    def test_lqr_usage_example_runs(self):
        doc = (REPO / "docs/full_state_feedback_controller.md").read_text()
        section = doc.split("### 11.1", 1)[1].split("### 11.2", 1)[0]
        example = re.search(r"~~~python\n(.*?)\n~~~", section, re.S).group(1)
        namespace = {}
        exec(compile(example, "LQR documentation example", "exec"), namespace)
        pwm = np.asarray(namespace["motor_pwm"])
        self.assertEqual(pwm.shape, (4,))
        self.assertTrue(np.all(np.isfinite(pwm)))
        self.assertTrue(np.all((pwm >= 0) & (pwm <= 65535)))


class TestPublicWheel(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("setuptools"), "wheel test needs setuptools>=77 (dev extra)")
    def test_wheel_runs_without_the_source_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            shutil.copytree(REPO / "crazyflie_sim", source / "crazyflie_sim",
                            ignore=shutil.ignore_patterns("__pycache__"))
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                shutil.copy2(REPO / name, source / name)
            build = subprocess.run(
                [sys.executable, "-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
                cwd=source, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            installed = root / "installed"
            with zipfile.ZipFile(next((source / "dist").glob("*.whl"))) as wheel:
                wheel.extractall(installed)
            probe = r'''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from crazyflie_sim import runsim
from crazyflie_sim.experiments.benchmark import suite
from crazyflie_sim.sim.config import load_config, params_overridden
from crazyflie_sim.sim.simulation import Simulation
assert Path(runsim.__file__).parent == Path(sys.argv[1]) / "crazyflie_sim"
cfg = runsim.load_run_config(runsim.parse_args([]))
assert cfg.scenario.name == "default"
cases = suite()
assert len(cases) == 7, cases
for path in cases.values():
    cfg = load_config(path)
    with params_overridden(cfg.params_overrides):
        run = Simulation.from_scenario(cfg.scenario, cfg.controller_mode, cfg.gains(), duration_s=0.004).run()
    assert len(run["t"]) == 2
'''
            result = subprocess.run(
                [sys.executable, "-I", "-c", probe, str(installed)],
                cwd=root, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TestRecording(unittest.TestCase):
    def test_gif_duration_and_one_advance_per_frame(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image, ImageSequence
        from crazyflie_sim import runsim
        from crazyflie_sim.visualization import quad_plot

        self.addCleanup(plt.close, "all")
        calls = []
        original = runsim.make_frame_source

        def source(*args):
            advance = original(*args)

            def frame(i):
                payload = advance(i)
                calls.append((i, payload["time"]))
                return payload

            return frame

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runsim, "make_frame_source", side_effect=source), \
                mock.patch.object(quad_plot.animation.writers, "is_available", side_effect=lambda name: name == "pillow"):
            path = Path(tmp) / "flight.gif"
            runsim.main([str(REPO / "crazyflie_sim/configs/default.yaml"), "--duration", "0.1", "--record", str(path)])
            with Image.open(path) as image:
                duration_ms = sum(frame.info["duration"] for frame in ImageSequence.Iterator(image))
        self.assertEqual(duration_ms, 100)
        self.assertEqual([i for i, _ in calls], list(range(5)))
        np.testing.assert_allclose([t for _, t in calls], [0.02, 0.04, 0.06, 0.08, 0.1])

    def test_mp4_encoder_uses_requested_frame_rate(self):
        from crazyflie_sim.visualization import quad_plot

        animation = mock.Mock()
        with mock.patch.object(quad_plot.animation.writers, "is_available", return_value=True), \
                mock.patch.object(quad_plot.animation, "FFMpegWriter") as writer:
            quad_plot._save_animation(animation, "flight.mp4", fps=50)
        writer.assert_called_once_with(fps=50, codec="libx264")
        animation.save.assert_called_once_with("flight.mp4", dpi=80, writer=writer.return_value)


if __name__ == "__main__":
    unittest.main()
