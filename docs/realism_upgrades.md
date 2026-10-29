# Realism Upgrades — Design & Usage Notes

This note documents the config-switchable realism features added on top of the
default simulator. **Every feature here is OFF under default settings**, so the
default numerical behavior is unchanged. That invariant is enforced by
`tests/test_golden.py` (eleven short closed-loop scenarios).

All features are driven from config files (`configs/*.yaml`, keys documented
in `crazyflie_sim/sim/config.py`); no module constants need to be edited to run an
experiment.

---

## 1. Sensor models (`crazyflie_sim/model/sensors.py`)

### IMU

- Gyro: `omega_meas = omega_true + b_g + n_g`, with a constant per-run bias
  `b_g ~ N(0, gyro_bias_std)` drawn once at construction and white noise
  `n_g ~ N(0, gyro_noise_std)` per call.
- Accelerometer: the specific force in the body frame — exactly the plant's
  non-gravity force per unit mass:

  ```
  a_meas = (F_thrust_body + R_world_to_body * (F_ext + F_aero)) / m  +  b_a + n_a
  F_thrust_body = [0, 0, sum(motor_thrusts) * ground_effect_factor]
  F_aero        = crazyflie_sim/model/aero.py force on the relative airflow v - v_wind
                  (legacy: -[k_xy, k_xy, k_z] * |v| * v)
  ```

  Gravity is excluded, as on a real accelerometer. Dryden gusts (below) appear
  in this signal via `F_ext`, which is the intended physics.

### Mocap

- Position + RPY attitude samples at a configurable `rate_hz`, with per-axis
  white noise and an integer `delay_steps` latency (ring buffer). The first
  sample is emitted at `t = 0` and primes the complementary-filter ablation.

### Lighthouse (`crazyflie_sim/model/lighthouse.py`)

- Two base stations and four deck photodiodes.  The measurement is the
  firmware sweep angle, not delayed Euler RPY.  Closed-loop hardware is
  Lighthouse V2, fixed in `crazyflie_sim/model/lighthouse.py`: `LH2_PROTOCOL`,
  `LH2_TILT_RAD = π/6`, `LH2_RATE_HZ = 50`, `LH2_DELAY_STEPS = 1`
  (~20 ms), `LH2_ANGLE_NOISE_STD` (milliradian jitter), and
  `LH2_ANGLE_BIAS_STD` (unmodeled per-run calibration bias sized so the
  navigation EKF's seed-0 Euclidean error versus plant truth is in the
  Taffanel 2021 Lighthouse accuracy class of about 2--4 cm).  Experiment
  YAML cannot override these.

### Gyro range

- `params.gyro_range_dps = 2000` is the firmware BMI088 full-scale.  A
  40 rad/s post-loss spin is about 2290 deg/s, so that hardware range
  saturates.
- `params.sim_gyro_range_dps = 4000` is the research-plant navigation IMU
  so the navigation EKF is not first limited by saturation.  The EKF still
  consumes whatever clipped measurement the IMU returns.

## 2. State estimator (`crazyflie_sim/estimation/estimator.py`, `crazyflie_sim/estimation/nav_ekf.py`)

`build_estimator(..., backend=kalman|complementary)` (default `kalman`).

`ErrorStateNavEKF` (error state: world pos, vel, body attitude error
`d0:2`, gyro bias; quaternion stored outside and injected after updates).

- Predict (500 Hz IMU): quaternion kinematics from bias-corrected gyro;
  specific force rotated by the estimated quaternion, then pos/vel
  integrate.  No `euler_rates`.
- Correct: lighthouse *scalar* sweep-angle updates (firmware
  `kalmanCoreScalarUpdate` structure) on the V2 ±30° planes.
  The closed-loop path does not apply a complementary triangulation snap.
- Delayed sweep angles are applied to a pose rotated back by
  ωτ; delayed Euler RPY is never a measurement.

`ComplementaryEstimator` remains the mocap-Euler ablation (pos, vel, RPY,
gyro bias, `euler_rates` prediction).

`EstimatedQuadView` wraps the real `Quadcopter`: state accessors return the
estimate (RPY via `RotToRPY`); everything else (`battery_voltage`,
`rotor_fault`, plot helpers) delegates to the truth.

Hook point: `crazyflie_sim.sim.simulation.Simulation(..., estimator=...)` (config key
`estimator:`) — the controller reads the view instead of the plant.  Without
an estimator the controller reads the plant itself.

Caution: `EstimatedQuadView` passes every attribute it does not override
(rotor speed, motor thrusts, rotor_fault) straight to the true plant, so a
controller can read ground truth without noticing.

**Common random numbers (CRN):** `crazyflie_sim.experiments.compare` builds one fresh
estimator per controller with the SAME seed (`seed`, default 0), so every
controller sees the identical noise realization — a paired comparison.
Under Monte Carlo, replicate `i` uses `seed + i` (independent across
replicates, CRN kept across controllers within a replicate).

## 3. Gust / turbulence wind (`crazyflie_sim/model/wind.py`, mode `dryden`)

Per-axis discrete Ornstein–Uhlenbeck (first-order Dryden-style) process,
advanced once per new time value passed to `WindModel.value(t)`:

```
phi = exp(-dt / tau);   x <- phi * x + sigma * sqrt(1 - phi^2) * xi,  xi ~ N(0,1)
F_axis(t) = offset + x
```

```yaml
wind:
  dryden_seed: 42
  x: {mode: dryden, dryden_sigma: 0.15, dryden_tau: 1.0, offset: 0.0}
```

- Each `WindModel` owns its RNG (`dryden_seed`), so runs never share gust
  state; draws are in fixed x, y, z order and the gust trajectory is a pure
  function of (seed, call times).
- Cache/reset semantics: a repeated call at the same `t` returns the cached
  value; an earlier `t` restarts the gusts; gated periods (`t < start_time`)
  do not advance the state.  `Simulation` calls `reset()` at the start of
  every run.

Also available: `smoothstep` mode — `F = mag * h^2 (3 - 2h)`,
`h = clip(t_rel / rise, 0, 1)` — used by the benchmark gust (`crazyflie_sim/configs/benchmark/_gust.yaml`).

## 4. Battery voltage sag (`crazyflie_sim/model/params.py` + `crazyflie_sim/model/quadcopter.py`)

Off by default (`battery_sag_enabled = False`). When enabled:

```
i_est    = (sum(motor_thrusts) / (4 * thrust_cmd_max_per_motor)) * battery_max_current_draw
v_target = clip(OCV - i_est * battery_internal_resistance, battery_voltage_min, battery_voltage_max)
V       += (dt / battery_sag_tau) * (v_target - V)
```

`OCV` is `battery_voltage_nominal` (or the `Quadcopter(battery_voltage=...)`
ctor kwarg). The sagged `battery_voltage` feeds the existing
voltage-compensated PWM↔thrust maps each tick. Enable via
`params_overrides: {battery_sag_enabled: true}`.

## 5. Aerodynamic model v2 (`crazyflie_sim/model/aero.py`)

Three independent switches (all default OFF; with everything off the drag
computation is bit-identical to the historical inline expression):

- **Rotor-induced drag** (`params_overrides: {rotor_drag_enabled: true}`):
  `F = -[dx, dy, 0] * v_rel_body` in the body frame — the dominant horizontal
  damping on nano-quadrotors (~0.0956 N·s/m from momentum theory at hover,
  ~15-20x the parasitic drag at 1 m/s, LINEAR in velocity). Coefficients:
  `rotor_drag_coefficient_{x,y}`. Identify from hardware when possible; the
  physical coefficient scales ~sqrt(thrust), the constant value is the
  standard simplification (Faessler et al., arXiv:1604.07797).
- **Body-frame parasitic drag** (`params_overrides: {body_frame_drag_enabled: true}`):
  the `aero_drag_factor_{xy,z}` factors act componentwise on the BODY-frame
  relative airflow (`-k*|v_i|*v_i`) instead of the world frame with `|v|`
  coupling — matters once the attitude leaves the hover neighborhood.
- **Ground effect** (`params_overrides: {ground_effect_enabled: true}`):
  realized motor thrusts are multiplied by `1/(1 - (R/4z))^2`
  (Cheeseman & Bennett), saturated at 4/3 for `z < R/2` (`prop_radius` =
  0.0275 m for the 55 mm CF 2.1 Brushless propellers). The IMU model reads
  `quad.ground_effect_factor`, so the accelerometer sees the same augmented
  thrust the plant applies.

All aero forces act on the relative airflow `v_rel = v - v_wind`, and the
IMU's specific force mirrors the plant's force terms through the same
`crazyflie_sim/model/aero.py` functions, so plant and accelerometer never disagree.

## 6. Wind as a velocity field (`wind: {as_velocity: true}`)

```yaml
wind:
  as_velocity: true          # waveform values are now m/s, not N
  x: {mode: constant, constant: 2.0}   # m/s steady wind in +x
```

- The waveform machinery (modes, OU gusts, gating, seeding, caching) is the
  same; only the units (m/s) and the force pathway change: the value feeds
  the aero model as `v_wind`, and no external force is applied, so the wind
  is not double-counted.
- Physical consequences: a steady wind produces a steady drag offset force
  (rotor drag ≈ `dx * v_wind` when enabled), gusts enter the accelerometer
  through the aero forces exactly as on the real vehicle, and the drag model
  responds to wind (it cannot in force mode).

## 7. Monte Carlo parameter uncertainty (`crazyflie_sim/experiments/monte_carlo.py`)

Run with `python -m crazyflie_sim.experiments.compare CONFIG --monte-carlo`.

Config block (validated by `validate_monte_carlo_spec`):

```yaml
monte_carlo:
  n_runs: 10
  seed: 7
  perturb:
    mass: {dist: uniform, low: 0.038, high: 0.042}        # absolute (default)
    I:    {dist: uniform, low: 0.95, high: 1.05, mode: scale}  # scale-only; 3 draws on the diagonal
    aero_drag_factor_xy: {dist: normal, mean: 0.00539, std: 0.000539, clip: [0.0043, 0.0065]}
    motor_tau_up: {dist: uniform, low: 0.0375, high: 0.0625}
    battery_voltage_nominal: {dist: uniform, low: 3.7, high: 4.0}
```

- Perturb keys are restricted to the `PARAMS_OVERRIDE_KEYS` whitelist in
  `crazyflie_sim/sim/config.py`; application goes through `params_overridden` (incl. the
  `I` → `invI` consistency rule) and is restored after each replicate.
- One `default_rng(seed)` draws all replicates in order ⇒ identical
  perturbation sequences for identical configs.
- Outputs in the run dir: `monte_carlo_report.json` (validated spec,
  per-replicate overrides + metrics, aggregate) and figures/time series for
  replicate 0 (`replicate_0/`).

## 8. Control link delay / jitter / dropout (`crazyflie_sim/model/control_link.py`)

Off by default. Enable with:

```yaml
control_link:
  enabled: true
  delay_steps: 2
  dropout_probability: 0.02
  jitter_std_steps: 0.5
  seed: 0
```

Hooked after `controller.run` and before `quad.update` in
`crazyflie_sim.sim.simulation.Simulation.step`. When disabled the PWM path is unchanged.

## 9. Quaternion attitude (`params.use_quaternion_attitude`)

Off by default. When enabled, `Quadcopter.update` integrates translation +
body rates + a unit quaternion with RK4; `state[6:9]` mirrors RPY via
`quat_to_rpy` for the public API. Rotation matrices come from the quaternion
directly, so large-roll fault maneuvers do not hit the Euler singularity in
`euler_rates`.

## 10. Reproducibility contract

- Same config ⇒ same run: deterministic plant/controllers; all randomness
  (gusts, sensor noise, MC sampling, control-link RNG) is seeded through the
  config.
- Every archived run (`$SIM_RESULTS_DIR` or `results/runs/<ts>_<name>/`)
  carries `config.yaml` (resolved snapshot), `original_config.yaml`,
  `env.json` (git commit/branch/dirty, versions), the time series, and
  metrics.
- `tests/test_golden.py` guards the default behavior against regressions
  introduced by any of the above.
