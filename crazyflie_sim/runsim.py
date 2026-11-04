"""Fly one scenario with one controller and watch it live.

    python runsim.py                                   # default.yaml
    python runsim.py hover_wind_default.yaml           # a bundled config, or any path
    python runsim.py hover_wind_default.yaml --controller mellinger
    python runsim.py --sim-mode trajectory --controller full_state_feedback
    python runsim.py --gui                            # interactive experiment panel
    python runsim.py CONFIG --record sim.mp4           # write a video, no window
    python runsim.py CONFIG --archive                  # save the time series

The live view and the headless runs (experiments/, sim.simulation.run) step
the same Simulation, so a config flies identically in both.
"""

import argparse
import atexit
from pathlib import Path

import numpy as np

from crazyflie_sim.controllers import registry
import crazyflie_sim.model.params as params
from crazyflie_sim.sim import archive
from crazyflie_sim.sim.config import apply_params_overrides, load_config
from crazyflie_sim.sim.scenario import SIM_MODES, fault_case_text
from crazyflie_sim.sim.simulation import Simulation

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"
ANIMATION_HZ = 50  # GUI frames per second of simulated time
HISTORY_WINDOW_S = None  # None keeps the whole path; a float keeps a rolling window


def parse_args(argv=None):
    # No fixed prog: "runsim.py" from a checkout, "crazyflie-sim" when installed.
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", nargs="?", help="YAML/JSON run config: a path or the name of a bundled config "
                        "in crazyflie_sim/configs/ (default: default.yaml)")
    parser.add_argument("--config", dest="config_flag", metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument("--controller", choices=registry.available(), help="override controller_mode")
    parser.add_argument("--sim-mode", choices=SIM_MODES, help="override sim_mode")
    parser.add_argument("--duration", type=float, help="override duration_s (s)")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="open the interactive experiment panel")
    display.add_argument("--record", metavar="PATH", help="render to PATH (.mp4 or .gif) with matplotlib")
    parser.add_argument("--archive", action="store_true", help="save the run under the results directory")
    args = parser.parse_args(argv)
    if args.gui and args.archive:
        parser.error("--archive is for direct runs; use --gui to edit and save a run config")
    return args


def load_run_config(args):
    cfg = load_config(args.config or args.config_flag or DEFAULT_CONFIG)
    if args.controller:
        cfg.controller_mode = args.controller
    if args.sim_mode:
        cfg.scenario.sim_mode = args.sim_mode
    if args.duration:
        cfg.scenario.duration_s = args.duration
    return cfg


def world_vertical_spin_dps(quad):
    """World-vertical angular velocity in deg/s, not the Euler yaw derivative.

    The legacy payload key ``yaw_rate_dps`` carries this component. Nonzero
    values can describe back-and-forth motion without sustained spinning.
    """
    omega_world = np.asarray(quad.rotation_body_to_world(), dtype=float).dot(
        np.asarray(quad.omega(), dtype=float).reshape(3,)
    )
    return float(np.degrees(omega_world[2]))


def controller_status(backend):
    """(status_label, status_value, effectiveness_estimate) for the HUD."""
    if not hasattr(backend, "diagnostics"):
        return "", float("nan"), None
    diagnostics = backend.diagnostics()
    label = str(diagnostics.get("status_label", ""))
    value = float(diagnostics.get("status_value", float("nan")))
    estimate = diagnostics.get("effectiveness_estimate")
    if estimate is not None:
        estimate = np.asarray(estimate, dtype=float).reshape(4,)
    return label, value, estimate


def fault_payload(fault, t):
    if not fault.enabled:
        return {
            "rotor_fault_case": "",
            "rotor_fault_rotor": 0,
            "rotor_fault_rotors": [],
            "rotor_fault_start_time": float("nan"),
            "rotor_fault_state": "disabled",
        }
    rotors = list(fault.rotor_indices)
    return {
        "rotor_fault_case": fault.label,
        "rotor_fault_rotor": int(rotors[0]),
        "rotor_fault_rotors": rotors,
        "rotor_fault_start_time": float(fault.start_time_s),
        "rotor_fault_state": fault.state_label(t),
    }


def make_frame_source(sim, scenario, mode):
    """Return control_loop(i): advance one GUI frame and describe it."""
    steps_per_frame = int(round(1.0 / (ANIMATION_HZ * sim.dt)))
    wind = scenario.active_wind()
    wind_formula = wind.formula_text() if wind is not None else ""
    if scenario.sim_mode == "hover_wind":
        wait_until, wait_label = float(scenario.wind.start_time), "wind"
    elif scenario.sim_mode == "hover_rotor_fault":
        wait_until, wait_label = float(scenario.rotor_fault.start_time_s), "rotor fault"
    else:
        wait_until, wait_label = None, ""
    max_error = [float("nan")]

    def control_loop(i):
        for _ in range(steps_per_frame):
            if not sim.done:
                sim.step()
        quad, t = sim.quad, sim.t
        reference = sim.reference(t)
        error = float(np.linalg.norm(quad.position() - reference.pos))
        waiting = wait_until is not None and t < wait_until
        if not waiting and np.isfinite(error):
            max_error[0] = error if not np.isfinite(max_error[0]) else max(max_error[0], error)
        label, value, estimate = controller_status(sim.controller)
        payload = {
            "frame": quad.world_frame(),
            # Plant-integrated attitude, unwrapped: a post-loss spin keeps
            # accumulating yaw past +/-180 deg.
            "attitude_deg": np.degrees(quad.attitude()),
            "time": t,
            "wind_force_world": wind.value(t) if wind is not None else np.zeros(3),
            "wind_formula": wind_formula,
            "position_error_norm": error,
            "position_error_norm_for_max": error,
            "position_error_max": float(max_error[0]),
            "reference_pos": np.asarray(reference.pos, dtype=float).reshape(3,).copy(),
            "reference_yaw_deg": float(np.degrees(reference.yaw)),
            "yaw_rate_dps": world_vertical_spin_dps(quad),
            "waiting_for_wind": waiting,
            "position_error_waiting_label": wait_label if waiting else "",
            "controller_mode": mode,
            "controller_status_label": label,
            "controller_status_value": value,
            "motor_thrusts": np.asarray(quad.motor_thrusts, dtype=float).reshape(4,).copy(),
            # Ground truth, shown next to the controller's estimate.
            "effectiveness_truth": (
                np.asarray(quad.rotor_fault.effectiveness_vector(t), dtype=float)
                if quad.rotor_fault.enabled
                else None
            ),
            "effectiveness_estimate": estimate,
        }
        payload.update(fault_payload(quad.rotor_fault, t))
        return payload

    return control_loop


def view_config(sim, scenario):
    event_time, event_label = scenario.event()
    diagnostics = sim.controller.diagnostics() if hasattr(sim.controller, "diagnostics") else {}
    wind = scenario.active_wind()
    fault = scenario.active_fault()
    event_description = "Wind velocity" if wind is not None and wind.as_velocity else "Wind force"
    if fault.enabled:
        event_description = "Rotor {}: {}".format(
            ", ".join(map(str, fault.rotor_indices)), fault_case_text(fault.label)
        )
    return {
        "duration_s": sim.duration_s,
        "event_description": event_description if event_time is not None else "",
        "sim_mode": scenario.sim_mode,
        "wind_as_velocity": bool(wind is not None and wind.as_velocity),
        "wind_formula": wind.formula_text() if wind is not None else "",
        "event_time_s": event_time,
        "event_label": event_label,
        # Fault mode plots estimate + truth when the controller has an estimate.
        "rotor_plot": "effectiveness" if diagnostics.get("effectiveness_estimate") is not None else "thrust",
        "tracking": scenario.sim_mode == "trajectory",
        "reference_path": reference_path(sim),
        # Plant hover thrust per rotor: the propellers' display speed is
        # relative to it.
        "hover_thrust_n": float(sim.quad.mass * params.g / 4.0),
    }


def reference_path(sim, rate_hz=ANIMATION_HZ):
    """(N, 3) reference positions sampled over the whole run."""
    times = np.linspace(0.0, sim.duration_s, int(np.ceil(sim.duration_s * rate_hz)) + 1)
    return np.array([np.asarray(sim.reference(t).pos, dtype=float).reshape(3,) for t in times])


def main(argv=None):
    args = parse_args(argv)
    cfg = load_run_config(args)
    if args.gui:
        from crazyflie_sim.visualization.launcher import launch

        return launch(cfg)
    apply_params_overrides(cfg.params_overrides)
    scenario = cfg.scenario
    sim = Simulation.from_scenario(scenario, cfg.controller_mode, cfg.gains())
    print(
        "{}: {} on {} ({}, {:.1f} s, profile={}, plant={})".format(
            cfg.source,
            cfg.controller_mode,
            scenario.name,
            scenario.sim_mode,
            sim.duration_s,
            scenario.actuator_profile or "default",
            scenario.actuator_plant_backend or "default",
        )
    )

    if args.archive:
        run_dir = archive.make_run_dir(cfg.name)

        def save():
            archive.write_metadata(run_dir, cfg.to_dict(), extra={"entrypoint": "runsim.py"},
                                   source_config_path=cfg.source)
            archive.save_timeseries(run_dir / "timeseries.npz", sim.results())
            print("Archived run to", run_dir)

        atexit.register(save)

    _, plot_waypoints, _ = scenario.reference()
    frames = int(np.ceil(sim.duration_s * ANIMATION_HZ))
    control_loop = make_frame_source(sim, scenario, cfg.controller_mode)
    if args.record:
        from crazyflie_sim.visualization.quad_plot import plot_quad_3d

        plot_quad_3d(plot_waypoints, control_loop, frames=frames, history_window_s=HISTORY_WINDOW_S,
                     fps=ANIMATION_HZ, view_config=view_config(sim, scenario), save_path=args.record)
    else:
        from crazyflie_sim.visualization.quad_plot_pg import plot_quad_3d

        plot_quad_3d(plot_waypoints, control_loop, frames=frames, history_window_s=HISTORY_WINDOW_S,
                     fps=ANIMATION_HZ, view_config=view_config(sim, scenario))


if __name__ == "__main__":
    main()
