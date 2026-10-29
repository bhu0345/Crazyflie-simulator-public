"""The README's example controller (examples/my_ctrl.py) flies, and the README shows it."""

import importlib.util
import re
import textwrap
import unittest
from pathlib import Path

from crazyflie_sim.controllers import registry
from crazyflie_sim.experiments.benchmark import fly, suite

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "my_ctrl.py"


class TestExampleController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry.available()  # discover the built-in controllers first
        spec = importlib.util.spec_from_file_location("example_my_ctrl", EXAMPLE)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))  # registers "my_ctrl"

    @classmethod
    def tearDownClass(cls):
        registry._REGISTRY.pop("my_ctrl", None)

    def test_it_holds_the_gust_and_tracks_the_path(self):
        for name, limit_m in (("hover_gust", 0.2), ("trajectory", 0.02)):
            with self.subTest(scenario=name):
                _, metrics = fly(suite()[name], "my_ctrl")
                self.assertTrue(metrics["survived"])
                self.assertLess(metrics["rms_err_m"], limit_m)

    def test_readme_shows_the_example_verbatim(self):
        readme = (REPO / "README.md").read_text()
        section = readme.split("## Add a controller", 1)[1].split("\n## ", 1)[0]
        shown = textwrap.dedent(re.search(r"```python\n(.*?)```", section, re.S).group(1))
        code = EXAMPLE.read_text().split('"""', 2)[2].lstrip("\n")  # without the docstring
        self.assertEqual(shown.strip(), code.strip())


if __name__ == "__main__":
    unittest.main()
