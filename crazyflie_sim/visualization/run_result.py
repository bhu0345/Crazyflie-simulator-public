"""A finished GUI run, independent of the currently selected configuration."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from crazyflie_sim.sim import archive
from crazyflie_sim.sim.config import params_overridden
from crazyflie_sim.sim.metrics import compute_metrics


@dataclass
class RunResult:
    config: dict
    run: dict
    metrics: dict
    interval: dict
    state: str
    stop_time_s: float
    error: str = ""

    @classmethod
    def capture(cls, config, sim, state, error=""):
        raw = config.to_dict()
        if raw["sim_mode"] == "trajectory" and raw["waypoint_source"] == "csv":
            raw["waypoints_csv_path"] = str(Path(raw["waypoints_csv_path"]).resolve())
        run = {key: value.copy() if isinstance(value, np.ndarray) else value
               for key, value in sim.results().items()}
        event, _ = config.scenario.event()
        t = run["t"]
        reached = bool(len(t) and (event is None or t[-1] >= event))
        interval = {"start_s": 0.0 if event is None else float(event),
                    "end_s": float(t[-1]) if len(t) else None,
                    "event_reached": reached}
        metrics = {}
        if len(t):
            # The same override scope as the comparison CLI (e.g. PWM bounds).
            with params_overridden(config.params_overrides):
                metrics = compute_metrics(config.scenario, run, config.metrics)
            if not reached:
                # Existing CLI metrics retain their historical fallback to all
                # samples; the GUI explicitly reports the absent event window.
                metrics = {key: None if ("post_" in key or key.startswith(
                    ("rms_", "max_", "mean_", "steady_", "peak_", "settling_")))
                    else value for key, value in metrics.items()}
        return cls(raw, run, metrics, interval, state, float(sim.t), str(error))

    def summary(self):
        mode = self.config["sim_mode"]
        suffix = "_post_wind_m" if mode == "hover_wind" else "_post_fault_m"
        rms = "rms_err_m" if mode in ("trajectory", "hovering") else "rms_err" + suffix
        peak = "max_err_m" if mode in ("trajectory", "hovering") else "max_err" + suffix

        def reading(key):
            value = self.metrics.get(key)
            return "Not available" if value is None or not np.isfinite(value) else f"{value:.3f} m"

        interval = self.interval
        window = (f"{interval['start_s']:.3f}–{interval['end_s']:.3f} s"
                  if interval["end_s"] is not None and interval["event_reached"]
                  else "Event not reached" if interval["end_s"] is not None else "No samples")
        warnings = []
        if self.error:
            warnings.append("Run stopped: " + self.error)
        if self.metrics.get("all_finite") is False:
            warnings.append("Non-finite data recorded")
        if self.metrics.get("ground_contact"):
            warnings.append("Ground contact recorded")
        return (f"{self.state} · statistics: {window}\n"
                f"RMS {reading(rms)}    Peak {reading(peak)}    Final {reading('final_err_m')}\n"
                "Final error is the last recorded sample.\n" +
                ("\n".join(warnings) if warnings else "No data warnings recorded."))

    def export(self, root=None):
        run_dir = archive.make_run_dir(self.config["name"], root)
        try:
            archive.write_metadata(run_dir, self.config, extra={
                "entrypoint": "gui", "state": self.state,
                "stop_time_s": self.stop_time_s, "error": self.error or None,
            })
            archive.save_timeseries(run_dir / "timeseries.npz", self.run)
            (run_dir / "metrics.json").write_text(json.dumps(archive.archive_plain({
                "metrics": self.metrics, "interval": self.interval,
                "state": self.state, "stop_time_s": self.stop_time_s,
                "error": self.error or None,
            }), indent=2, allow_nan=False))
        except Exception as exc:
            raise OSError(f"Export failed in {run_dir}: {exc}") from exc
        return run_dir
