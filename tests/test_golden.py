"""Short reference flights must not change silently (see tools/golden.py)."""

import unittest

from crazyflie_sim.tools import golden


class TestGoldenTrajectories(unittest.TestCase):
    def test_every_golden_scenario_matches(self):
        for name in golden.scenarios():
            with self.subTest(scenario=name):
                problems = golden.compare(name)
                self.assertEqual(problems, [], "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
