"""Scalar metrics of a finished run (used by experiments/).

wind_rejection_metrics: after the wind onset; fault_tolerance_metrics: after
the rotor fault; iae / control_energy / pwm_saturation_ratio: any run.
"""

import numpy as np

import crazyflie_sim.model.params as params
from crazyflie_sim.sim.scenario import DT


def wind_rejection_metrics(
    t,
    pos,
    err,
    att,
    setpoint,
    event_time_s,
    dt,
    settling_threshold_m=0.10,
    settling_hold_s=2.0,
    steady_state_window_s=10.0,
) -> dict:
    """Wind-rejection metrics.

    All 'post-wind' metrics are computed only for t >= event_time_s to exclude
    the clean takeoff/hover transient before the disturbance begins.
    """
    t = np.asarray(t, dtype=float)
    e = np.asarray(err, dtype=float)
    pos = np.asarray(pos, dtype=float)
    att = np.asarray(att, dtype=float)
    sp = np.asarray(setpoint, dtype=float).reshape(3)
    dt = float(dt)

    t_wind = float(event_time_s)
    mask_post = t >= t_wind
    if not np.any(mask_post):
        mask_post = np.ones_like(t, dtype=bool)

    e_post = e[mask_post]
    t_post = t[mask_post]

    # Steady-state window: last steady_state_window_s seconds of the whole run
    t_max = float(t[-1])
    mask_ss = t >= max(t_wind, t_max - steady_state_window_s)
    if not np.any(mask_ss):
        mask_ss = mask_post
    e_ss = e[mask_ss]

    # ---- settling time -------------------------------------------------------
    # First time (within the post-wind interval) at which the error drops below
    # settling_threshold_m and stays there for settling_hold_s seconds.
    settling_time = float("nan")
    hold_steps = max(1, int(np.ceil(settling_hold_s / dt)))
    for i in range(len(e_post)):
        if e_post[i] < settling_threshold_m:
            end_i = min(i + hold_steps, len(e_post))
            window = e_post[i:end_i]
            if len(window) == hold_steps and np.all(window < settling_threshold_m):
                settling_time = float(t_post[i])
                break

    # ---- spatial drift -------------------------------------------------------
    dx = pos[:, 0] - sp[0]
    dy = pos[:, 1] - sp[1]
    dz = pos[:, 2] - sp[2]
    xy_drift = np.hypot(dx, dy)
    post_xy = xy_drift[mask_post]
    post_abs_dz = np.abs(dz[mask_post])

    # ---- attitude excursion under disturbance --------------------------------
    post_roll_deg = np.abs(np.degrees(att[mask_post, 0]))
    post_pitch_deg = np.abs(np.degrees(att[mask_post, 1]))

    return {
        "max_err_post_wind_m": float(np.max(e_post)),
        "rms_err_post_wind_m": float(np.sqrt(np.mean(e_post**2))),
        "mean_err_post_wind_m": float(np.mean(e_post)),
        "steady_state_rms_m": float(np.sqrt(np.mean(e_ss**2))),
        "steady_state_mean_m": float(np.mean(e_ss)),
        "max_xy_drift_m": float(np.max(post_xy)),
        "max_z_drift_m": float(np.max(post_abs_dz)),
        "settling_time_s": settling_time,
        "peak_roll_deg": float(np.max(post_roll_deg)),
        "peak_pitch_deg": float(np.max(post_pitch_deg)),
        "final_err_m": float(e[-1]),
    }


def fault_tolerance_metrics(t, pos, err, omega, setpoint, fault_start_time_s) -> dict:
    """Rotor-fault tolerance metrics."""
    t = np.asarray(t, dtype=float)
    pos = np.asarray(pos, dtype=float)
    err = np.asarray(err, dtype=float)
    omega = np.asarray(omega, dtype=float)
    sp = np.asarray(setpoint, dtype=float).reshape(3)

    mask_post = t >= float(fault_start_time_s)
    if not np.any(mask_post):
        mask_post = np.ones_like(t, dtype=bool)

    e_post = err[mask_post]
    xy_drift = np.hypot(pos[:, 0] - sp[0], pos[:, 1] - sp[1])
    altitude_loss = np.maximum(sp[2] - pos[:, 2], 0.0)
    yaw_rate_dps = np.degrees(omega[:, 2])

    ground_hits = np.flatnonzero(mask_post & (pos[:, 2] <= 1.0e-6))
    time_to_ground_s = float("nan")
    if ground_hits.size > 0:
        time_to_ground_s = float(t[ground_hits[0]])

    return {
        "max_err_post_fault_m": float(np.max(e_post)),
        "rms_err_post_fault_m": float(np.sqrt(np.mean(e_post ** 2))),
        "max_xy_drift_m": float(np.max(xy_drift[mask_post])),
        "max_altitude_loss_m": float(np.max(altitude_loss[mask_post])),
        "time_to_ground_s": time_to_ground_s,
        "peak_yaw_rate_dps": float(np.max(np.abs(yaw_rate_dps[mask_post]))),
        "final_err_m": float(err[-1]),
    }


# ---------------------------------------------------------------------------
# Additive helpers (only ever ADD keys to metrics payloads)
# ---------------------------------------------------------------------------


def iae(t, err, event_time_s) -> float:
    """Integral of absolute-error equivalent: trapezoidal integral of ``err``
    over t >= event_time_s."""
    t = np.asarray(t, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = t >= float(event_time_s)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    return float(np.trapezoid(err[mask], t[mask]))


def control_energy(t, motor_pwm) -> float:
    """Sum over steps of ||pwm||^2 * dt (dt from the recorded time grid)."""
    t = np.asarray(t, dtype=float)
    pwm = np.asarray(motor_pwm, dtype=float)
    if t.size == 0 or pwm.size == 0:
        return 0.0
    dts = np.diff(t, prepend=0.0)
    return float(np.sum(np.sum(pwm**2, axis=1) * dts))


def pwm_saturation_ratio(motor_pwm, pwm_min, pwm_max) -> float:
    """Fraction of (step, motor) entries within 1e-6 of pwm_min or pwm_max."""
    pwm = np.asarray(motor_pwm, dtype=float)
    if pwm.size == 0:
        return 0.0
    saturated = (pwm <= float(pwm_min) + 1e-6) | (pwm >= float(pwm_max) - 1e-6)
    return float(np.mean(saturated))


def compute_metrics(scenario, run, metric_options=None):
    """Scalar metrics for one run, chosen by the scenario's sim_mode."""
    options = dict(metric_options or {})
    t, pos, err = run["t"], run["pos"], run["err"]
    setpoint = np.asarray(scenario.hover_setpoint, dtype=float)
    event_time, _ = scenario.event()
    event_time = 0.0 if event_time is None else event_time
    if scenario.sim_mode == "hover_wind":
        out = wind_rejection_metrics(
            t, pos, err, run["att"], setpoint, event_time, DT, **options
        )
    elif scenario.sim_mode == "hover_rotor_fault":
        out = fault_tolerance_metrics(t, pos, err, run["omega"], setpoint, event_time)
    else:
        out = {
            "rms_err_m": float(np.sqrt(np.mean(np.asarray(err) ** 2))),
            "max_err_m": float(np.max(err)),
            "final_err_m": float(err[-1]),
        }
    out["iae"] = iae(t, err, event_time)
    out["control_energy"] = control_energy(t, run["motor_pwm"])
    out["pwm_saturation_ratio"] = pwm_saturation_ratio(
        run["motor_pwm"], params.pwm_min, params.pwm_max
    )
    out["all_finite"] = bool(
        all(np.all(np.isfinite(run[k])) for k in ("pos", "vel", "att", "omega", "motor_pwm"))
    )
    airborne = np.asarray(t) >= max(event_time, 1.0)  # skip a takeoff from z = 0
    out["ground_contact"] = bool(np.any(np.asarray(pos)[airborne, 2] <= 0.0))
    out["wall_time_s"] = float(run["wall_time_s"])
    return out
