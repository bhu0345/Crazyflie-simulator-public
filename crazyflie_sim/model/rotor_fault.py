from collections.abc import Iterable
from dataclasses import dataclass, field

import crazyflie_sim.model.params as params


@dataclass
class RotorFaultConfig:
    """Time-gated rotor effectiveness fault.

    ``rotor_index`` accepts either one 1-based rotor index or an iterable of
    1-based rotor indices. For backward compatibility, the public
    ``rotor_index`` attribute remains an integer when exactly one rotor is
    configured.
    """

    enabled: bool = False
    rotor_index: int = 1
    start_time_s: float = 0.0
    effectiveness: float = 1.0
    label: str = ""
    mechanism: str = "motor_loe"
    rotor_indices: tuple[int, ...] = field(init=False)
    start_times_s: tuple[float, ...] = field(init=False)

    def __post_init__(self):
        self.enabled = bool(self.enabled)
        self.rotor_indices = self._normalize_rotor_indices(self.rotor_index)
        self.rotor_index = (
            self.rotor_indices[0]
            if len(self.rotor_indices) == 1
            else list(self.rotor_indices)
        )
        self.start_times_s = self._normalize_start_times(
            self.start_time_s, len(self.rotor_indices)
        )
        # Backward-compatible scalar: the first onset of a scheduled fault.
        self.start_time_s = min(self.start_times_s)
        self.effectiveness = float(self.effectiveness)
        self.mechanism = str(self.mechanism).lower()
        self.label = str(self.label)

        for rotor_index in self.rotor_indices:
            if rotor_index < 1 or rotor_index > 4:
                raise ValueError("rotor_index entries must be in 1..4.")
        if self.effectiveness < 0.0 or self.effectiveness > 1.0:
            raise ValueError("effectiveness must be in [0, 1].")
        if self.mechanism not in ("motor_loe", "prop_loe", "combined"):
            raise ValueError(
                "mechanism must be 'motor_loe', 'prop_loe', or 'combined'."
            )

    @staticmethod
    def _normalize_start_times(start_time_s, count):
        if isinstance(start_time_s, Iterable) and not isinstance(
            start_time_s, (str, bytes)
        ):
            values = tuple(float(value) for value in start_time_s)
            if len(values) == 1 and count > 1:
                values = values * count
            if len(values) != count:
                raise ValueError(
                    "start_time_s must be scalar or match rotor_index length."
                )
        else:
            values = (float(start_time_s),) * count
        if any(value < 0.0 for value in values):
            raise ValueError("start_time_s entries must be nonnegative.")
        return values

    @staticmethod
    def _normalize_rotor_indices(rotor_index):
        if isinstance(rotor_index, Iterable) and not isinstance(rotor_index, (str, bytes)):
            indices = tuple(int(index) for index in rotor_index)
        else:
            indices = (int(rotor_index),)
        if not indices:
            raise ValueError("rotor_index must include at least one rotor.")
        if len(set(indices)) != len(indices):
            raise ValueError("rotor_index entries must be unique.")
        return indices

    @property
    def rotor_slot(self):
        return self.rotor_slots[0]

    @property
    def rotor_slots(self):
        return tuple(rotor_index - 1 for rotor_index in self.rotor_indices)

    def is_active(self, sim_time):
        if not self.enabled:
            return False
        if sim_time is None:
            sim_time = 0.0
        return any(float(sim_time) >= onset for onset in self.start_times_s)

    def active_rotor_slots(self, sim_time=None):
        if not self.enabled:
            return ()
        if sim_time is None:
            sim_time = 0.0
        now = float(sim_time)
        return tuple(
            slot
            for slot, onset in zip(self.rotor_slots, self.start_times_s)
            if now >= onset
        )

    def effectiveness_vector(self, sim_time=None):
        values = [1.0] * 4
        for slot in self.active_rotor_slots(sim_time):
            values[slot] = self.effectiveness
        return tuple(values)

    def physical_scales(self, sim_time=None):
        """Return per-rotor ``(speed_scale, cT_scale)`` arrays.

        Each mechanism realizes the configured steady thrust effectiveness:
        motor LOE changes mechanical RPM, prop LOE changes cT at the same RPM,
        and combined shares the loss equally in the two thrust contributions.
        The controller never receives these mechanism-specific quantities.
        """
        speed = [1.0] * 4
        coefficient = [1.0] * 4
        if self.mechanism == "motor_loe":
            speed_factor = self.effectiveness ** 0.5
            coefficient_factor = 1.0
        elif self.mechanism == "prop_loe":
            speed_factor = 1.0
            coefficient_factor = self.effectiveness
        else:
            speed_factor = self.effectiveness ** 0.25
            coefficient_factor = self.effectiveness ** 0.5
        for slot in self.active_rotor_slots(sim_time):
            speed[slot] = speed_factor
            coefficient[slot] = coefficient_factor
        return tuple(speed), tuple(coefficient)

    def yaw_torque_coefficient_scales(self, sim_time=None):
        """Return plant-only propeller reaction-torque coefficient scales.

        Motor LOE preserves the nominal torque/thrust ratio through the lower
        rotor speed.  For propeller damage, the generic simulation model uses
        the independently configurable reaction-torque exponent: reaction
        torque still vanishes at complete loss, but is not forced to remain
        proportional to thrust.  The mismatch is deliberately absent from
        controller feedback.
        """
        coefficient = [1.0] * 4
        if self.mechanism == "motor_loe":
            coefficient_factor = 1.0
        elif self.mechanism == "prop_loe":
            coefficient_factor = self.effectiveness
        else:
            # combined uses kappa=sqrt(lambda)
            coefficient_factor = self.effectiveness ** 0.5
        factor = coefficient_factor ** float(
            params.prop_yaw_torque_loss_exponent
        )
        for slot in self.active_rotor_slots(sim_time):
            coefficient[slot] = factor
        return tuple(coefficient)

    def state_label(self, sim_time):
        if not self.enabled:
            return "disabled"
        return "active" if self.is_active(sim_time) else "inactive"


@dataclass
class PerRotorFaultConfig(RotorFaultConfig):
    """RotorFaultConfig with one effectiveness value per configured rotor."""

    effectiveness_per_rotor: tuple = ()
    restore_s: float = None      # absolute time at which every rotor returns to 1

    def __post_init__(self):
        values = tuple(float(v) for v in self.effectiveness_per_rotor)
        if values:
            self.effectiveness = min(values)
        super().__post_init__()
        if not values:
            values = (float(self.effectiveness),) * len(self.rotor_indices)
        if len(values) != len(self.rotor_indices):
            raise ValueError("effectiveness_per_rotor must match rotor_index length.")
        if any(v < 0.0 or v > 1.0 for v in values):
            raise ValueError("effectiveness_per_rotor entries must lie in [0, 1].")
        self.effectiveness_per_rotor = values

    def _active_values(self, sim_time=None):
        if not self.enabled:
            return {}
        now = 0.0 if sim_time is None else float(sim_time)
        if self.restore_s is not None and now >= float(self.restore_s):
            return {}
        return {
            slot: value
            for slot, onset, value in zip(
                self.rotor_slots, self.start_times_s, self.effectiveness_per_rotor
            )
            if now >= onset
        }

    def effectiveness_vector(self, sim_time=None):
        values = [1.0] * 4
        for slot, value in self._active_values(sim_time).items():
            values[slot] = value
        return tuple(values)

    def physical_scales(self, sim_time=None):
        speed = [1.0] * 4
        coefficient = [1.0] * 4
        for slot, value in self._active_values(sim_time).items():
            if self.mechanism == "motor_loe":
                speed[slot] = value ** 0.5
            elif self.mechanism == "prop_loe":
                coefficient[slot] = value
            else:
                speed[slot] = value ** 0.25
                coefficient[slot] = value ** 0.5
        return tuple(speed), tuple(coefficient)

    def yaw_torque_coefficient_scales(self, sim_time=None):
        coefficient = [1.0] * 4
        exponent = float(params.prop_yaw_torque_loss_exponent)
        for slot, value in self._active_values(sim_time).items():
            if self.mechanism == "motor_loe":
                base = 1.0
            elif self.mechanism == "prop_loe":
                base = value
            else:
                base = value ** 0.5
            coefficient[slot] = base ** exponent
        return tuple(coefficient)
