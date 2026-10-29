"""Actuator-side timing imperfections: delay, jitter, and packet dropouts.

All effects are OFF unless ``enabled`` is true. The default path returns the
controller command unchanged so golden trajectories stay bit-identical.
"""

from __future__ import annotations

from collections import deque

import numpy as np


CONTROL_LINK_KEYS = frozenset(
    [
        "enabled",
        "delay_steps",
        "dropout_probability",
        "jitter_std_steps",
        "seed",
    ]
)


def validate_control_link_spec(spec):
    """Validate and normalize a ``control_link:`` config dict."""
    if spec is None:
        return {"enabled": False}
    if not isinstance(spec, dict):
        raise TypeError("control_link must be a dict")
    unknown = set(spec) - CONTROL_LINK_KEYS
    if unknown:
        raise ValueError(
            "control_link unknown keys: {}".format(sorted(unknown))
        )
    out = {
        "enabled": bool(spec.get("enabled", False)),
        "delay_steps": int(spec.get("delay_steps", 0)),
        "dropout_probability": float(spec.get("dropout_probability", 0.0)),
        "jitter_std_steps": float(spec.get("jitter_std_steps", 0.0)),
        "seed": int(spec.get("seed", 0)),
    }
    if out["delay_steps"] < 0:
        raise ValueError("control_link.delay_steps must be >= 0")
    if not 0.0 <= out["dropout_probability"] <= 1.0:
        raise ValueError(
            "control_link.dropout_probability must be in [0, 1]"
        )
    if out["jitter_std_steps"] < 0.0:
        raise ValueError("control_link.jitter_std_steps must be >= 0")
    return out


class ControlChannel:
    """Fixed-rate command channel with optional delay / dropout / jitter.

    Delay is an integer FIFO of length ``delay_steps`` (plus a jittered extra
    hold drawn each tick when ``jitter_std_steps`` > 0). Dropouts reuse the
    last *applied* command without advancing the delay line's release.
    """

    def __init__(
        self,
        delay_steps=0,
        dropout_probability=0.0,
        jitter_std_steps=0.0,
        seed=0,
    ):
        self.delay_steps = int(delay_steps)
        self.dropout_probability = float(dropout_probability)
        self.jitter_std_steps = float(jitter_std_steps)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._queue = deque()
        self._last_applied = None
        self._hold_remaining = 0
        self.last_dropped = False
        self.last_delay_steps = self.delay_steps

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._queue.clear()
        self._last_applied = None
        self._hold_remaining = 0
        self.last_dropped = False
        self.last_delay_steps = self.delay_steps

    def apply(self, motor_pwm):
        """Return the PWM that should be sent to the plant this tick."""
        command = np.asarray(motor_pwm, dtype=float).reshape(4,).copy()

        if (
            self.dropout_probability > 0.0
            and self._last_applied is not None
            and self._rng.random() < self.dropout_probability
        ):
            self.last_dropped = True
            return self._last_applied.copy()
        self.last_dropped = False

        extra = 0
        if self.jitter_std_steps > 0.0:
            extra = int(
                max(
                    0,
                    round(float(self._rng.normal(0.0, self.jitter_std_steps))),
                )
            )
        effective_delay = self.delay_steps + extra
        self.last_delay_steps = effective_delay

        self._queue.append(command)
        while len(self._queue) > effective_delay + 1:
            self._queue.popleft()

        if len(self._queue) <= effective_delay:
            # Pipeline not yet full: hold last applied or the newest command.
            if self._last_applied is None:
                applied = command.copy()
            else:
                applied = self._last_applied.copy()
        else:
            applied = self._queue.popleft()

        self._last_applied = applied.copy()
        return applied


def build_control_link(spec):
    """Return a ``ControlChannel`` or ``None`` when the link is disabled."""
    cfg = validate_control_link_spec(spec)
    if not cfg["enabled"]:
        return None
    if (
        cfg["delay_steps"] == 0
        and cfg["dropout_probability"] == 0.0
        and cfg["jitter_std_steps"] == 0.0
    ):
        # Enabled but inert — still return a channel so diagnostics work.
        pass
    return ControlChannel(
        delay_steps=cfg["delay_steps"],
        dropout_probability=cfg["dropout_probability"],
        jitter_std_steps=cfg["jitter_std_steps"],
        seed=cfg["seed"],
    )
