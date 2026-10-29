# Contributing

Bug reports, new controllers, scenarios and model improvements are welcome.
Open an issue first for larger changes so the design can be discussed.

## Set up and test

```bash
pip install -e ".[gui,dev]"
python -m unittest discover -s tests -t .
```

The suite includes golden trajectories (`tests/golden/`), the documented
examples and a built-wheel check; it runs in well under a minute.  GUI tests
skip themselves when PyQt6, pyqtgraph or PyOpenGL is missing.

## How the code fits together

- **One step loop.** The live view, the experiments, the golden tool and the
  tests all advance the plant through `crazyflie_sim/sim/simulation.py:Simulation.step`.
  Add a hook to `Simulation` rather than a second loop.
- **One controller list.** A controller registers itself with
  `@register(...)` in `crazyflie_sim/controllers/`; nothing else lists controller modes.
  Take an `actuator_profile=None` argument and read the vehicle (mass,
  inertia, mixer, limits) from it, not from `crazyflie_sim/model/params.py`.
- **A run is a `Scenario`** (`crazyflie_sim/sim/scenario.py`), built from YAML by
  `crazyflie_sim/sim/config.py`.  Unknown config keys are errors.
- **`crazyflie_sim/model/` is the plant only**; estimators live in
  `crazyflie_sim/estimation/`.

## Changes that alter results

- If a change moves simulated trajectories on purpose, recapture the goldens
  with `python -m crazyflie_sim.tools.golden capture` (preferably in the conda environment
  of `environment.yml`) and say so in the pull request.  Otherwise
  `tests/test_golden.py` must stay green.
- `crazyflie_sim/configs/benchmark/` defines the published scores.  Do not edit a scenario
  there; add a new numbered file instead.  The seed policy and the reference
  controller are part of the score as well.

## Style

Match the surrounding code: its naming, comment density and idioms.  Keep
new options off by default so existing configs fly unchanged.
