import numpy as np
import scipy.integrate as integrate
from crazyflie_sim.utils.utils import (
    RPYToRot,
    quat_derivative,
    quat_to_rot_body_to_world,
    quat_to_rpy,
    rpy_to_quat,
)
import crazyflie_sim.model.actuators as actuators
import crazyflie_sim.model.aero as aero
import crazyflie_sim.model.params as params
from crazyflie_sim.model.actuator_profile import get_actuator_profile, graefe_speed, graefe_thrust
from crazyflie_sim.model.rotor_fault import RotorFaultConfig


LEGACY_THRUST_EULER = "legacy_thrust_euler"
ROTOR_SPEED_EXACT_ZOH = "rotor_speed_exact_zoh"
THRUST_EXACT_ZOH = "thrust_exact_zoh"
_ACTUATOR_PLANT_BACKENDS = frozenset(
    (LEGACY_THRUST_EULER, ROTOR_SPEED_EXACT_ZOH, THRUST_EXACT_ZOH)
)
BODY_DRAW_HEIGHT = 0.03  # m, the marker above the body centre in drawings


class Quadcopter:
    """Quadcopter class

    state  - 1 dimensional vector but used as 12 x 1:
             [x, y, z, xd, yd, zd, phi, theta, psi, p, q, r]
             where [phi, theta, psi] are roll, pitch, yaw and [p, q, r]
             are body-frame angular velocity components.
    motor_pwm - 4 x 1, 16-bit PWM command per rotor from controller

    The vehicle (mass, inertia, rotors, motors) is ``actuator_profile``; None
    is the default crazyflie21_physical.

    When ``params.use_quaternion_attitude`` is True at construction (or the
    per-instance override), attitude kinematics are integrated with a unit
    quaternion to avoid Euler singularities; ``state[6:9]`` still mirrors RPY
    for the public API.
    """

    def __init__(
        self,
        pos,
        attitude,
        battery_voltage=None,
        rotor_fault=None,
        battery_sag=None,
        use_quaternion=None,
        use_rotor_speed_model=None,
        mass=None,
        inertia=None,
        motor_tau_up=None,
        motor_tau_down=None,
        actuator_profile=None,
        actuator_plant_backend=None,
        thrust_map=None,
        rotor_inertia_kgm2=None,
        rotor_gyro=True,
        rotor_reaction=True,
        yaw_drag_scale=1.0,
    ):
        """pos = [x,y,z] attitude = [roll,pitch,yaw]

        battery_voltage sets the supply voltage (default
        params.battery_voltage_nominal); when the battery-sag model is
        enabled this value is the open-circuit voltage (OCV) and the
        terminal voltage sags below it under load.
        battery_sag=None reads params.battery_sag_enabled at construction;
        pass True/False to override for this instance.
        use_quaternion=None reads params.use_quaternion_attitude.
        actuator_profile accepts a shared ActuatorProfile or registry name
        (None: crazyflie21_physical).
        actuator_plant_backend accepts ``legacy_thrust_euler``,
        ``rotor_speed_exact_zoh``, or ``thrust_exact_zoh``.  When it is None,
        use_rotor_speed_model preserves its historical mapping: True selects
        the RPM backend and False selects the legacy thrust-Euler backend.
        mass/inertia/motor_tau_*/rotor_inertia_kgm2 are plant-only overrides
        of the profile used for robustness tests; the controller keeps the
        profile's values.
        """
        self.state = np.zeros(12)
        roll, pitch, yaw = attitude
        self.state[0] = pos[0]
        self.state[1] = pos[1]
        self.state[2] = max(0.0, pos[2])
        self.state[6] = roll
        self.state[7] = pitch
        self.state[8] = yaw
        self.battery_voltage = float(
            params.battery_voltage_nominal
            if battery_voltage is None
            else battery_voltage
        )
        # Optional first-order battery-sag model (default OFF). With sag
        # enabled, self._battery_ocv is the open-circuit voltage used by the
        # sag target in update().
        self._battery_ocv = self.battery_voltage
        self._battery_sag_enabled = (
            params.battery_sag_enabled if battery_sag is None else bool(battery_sag)
        )
        self._sag_voltage = self.battery_voltage
        self.rotor_fault = RotorFaultConfig() if rotor_fault is None else rotor_fault
        # Speed-to-thrust map of the ROTOR-SPEED plant.  'quadratic' is
        # f = c_T Omega^2 and the default.  'graefe_cubic' is the measured
        # curve of Graefe et al.; it replaces BOTH directions, the target-speed
        # inversion and the speed-to-thrust output, so that the healthy
        # steady-state PWM-to-thrust relation is unchanged and only the speed
        # at which a given thrust is produced moves.  Finite-amplitude
        # transients still differ, because the lag then acts on a different
        # quantity.  None takes the actuator profile's curve.
        self.actuator_profile = profile = get_actuator_profile(actuator_profile)
        self.thrust_map = str(profile.thrust_map if thrust_map is None else thrust_map)
        if self.thrust_map not in ("quadratic", "graefe_cubic"):
            raise ValueError("thrust_map must be 'quadratic' or 'graefe_cubic'")
        self._use_quaternion = (
            params.use_quaternion_attitude
            if use_quaternion is None
            else bool(use_quaternion)
        )
        if actuator_plant_backend is None:
            use_rpm = (
                params.rotor_speed_model_enabled
                if use_rotor_speed_model is None
                else bool(use_rotor_speed_model)
            )
            actuator_plant_backend = (
                ROTOR_SPEED_EXACT_ZOH if use_rpm else LEGACY_THRUST_EULER
            )
        actuator_plant_backend = str(actuator_plant_backend).strip().lower()
        if actuator_plant_backend not in _ACTUATOR_PLANT_BACKENDS:
            choices = ", ".join(sorted(_ACTUATOR_PLANT_BACKENDS))
            raise ValueError(
                "actuator_plant_backend must be one of: {}".format(choices)
            )
        # An explicit backend is authoritative.  This lets new configurations
        # coexist with old runners that still pass use_rotor_speed_model.
        self.actuator_plant_backend = actuator_plant_backend
        self._use_rotor_speed_model = (
            actuator_plant_backend == ROTOR_SPEED_EXACT_ZOH
        )
        self._plant_command_lower = profile.command_lower
        self._plant_command_upper = profile.command_upper
        self._rotor_speed_limit = profile.rotor_speed_limit
        self.mass = float(profile.mass if mass is None else mass)
        inertia_value = profile.inertia if inertia is None else inertia
        inertia_array = np.asarray(inertia_value, dtype=float)
        if inertia_array.shape == (3,):
            inertia_array = np.diag(inertia_array)
        self.inertia = inertia_array.reshape(3, 3).copy()
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.angular_damping = np.asarray(profile.angular_damping, dtype=float).reshape(3,)
        self.yaw_drag_coefficient = float(profile.yaw_drag_coefficient)
        self.motor_tau_up = float(
            profile.tau_up if motor_tau_up is None else motor_tau_up
        )
        self.motor_tau_down = float(
            profile.tau_down if motor_tau_down is None else motor_tau_down
        )
        if self.mass <= 0.0 or min(self.motor_tau_up, self.motor_tau_down) <= 0.0:
            raise ValueError("plant mass and motor time constants must be positive")
        if np.min(np.linalg.eigvalsh(0.5 * (self.inertia + self.inertia.T))) <= 0.0:
            raise ValueError("plant inertia must be positive definite")
        self._nominal_thrust_coefficient = float(profile.nominal_thrust_coefficient)
        self.quat = rpy_to_quat(float(roll), float(pitch), float(yaw))
        # Rotor angular momentum h = J_m sum_i s_i Omega_i about body z, with the
        # spin direction s = -sign(G[3, :]) of the yaw row.  The gyroscopic
        # moment -omega x (h e3) and the reaction moment -J_m sum_i s_i dOmega_i/dt
        # (dOmega/dt the speed change over the plant step) enter the realised
        # moment.  J_m = 0 leaves every operation of the plant unchanged.
        # None takes the profile's J_m.
        self.rotor_inertia_kgm2 = float(
            profile.rotor_inertia if rotor_inertia_kgm2 is None else rotor_inertia_kgm2
        )
        self.rotor_gyro = bool(rotor_gyro)
        self.rotor_reaction = bool(rotor_reaction)
        if self.rotor_inertia_kgm2 < 0.0:
            raise ValueError("rotor_inertia_kgm2 must be nonnegative")
        if self.rotor_inertia_kgm2 > 0.0:
            wrench = np.asarray(profile.wrench_matrix, dtype=float)
            self._rotor_signs = -np.sign(wrench.reshape(4, 4)[3, :])
        self._rotor_step = dict(prev=None, dt=None)
        if float(yaw_drag_scale) != 1.0:
            self.yaw_drag_coefficient = self.yaw_drag_coefficient * float(yaw_drag_scale)

        # Start motors close to hover thrust to avoid an unrealistic takeoff transient.
        initial_forces = np.clip(
            np.full(4, self.mass * params.g / 4.0, dtype=float),
            self._plant_command_lower,
            self._plant_command_upper,
        )
        initial_thrust_command = actuators.motor_forces_to_thrust_command(
            initial_forces, actuator_profile=profile
        )
        hover_pwm = actuators.cap_motor_pwm(
            actuators.thrust_command_to_pwm(
                initial_thrust_command,
                self.battery_voltage,
                actuator_profile=profile,
            )
        )
        self.motor_pwm = hover_pwm.copy()
        self.motor_thrusts = actuators.pwm_to_motor_forces(
            hover_pwm,
            self.battery_voltage,
            actuator_profile=profile,
        )
        # Initialise the speed state with the same map the plant will use, so a
        # cubic-map run does not start from a 20% thrust error.
        if self.thrust_map == "graefe_cubic":
            self.rotor_speeds = graefe_speed(self.motor_thrusts)
        else:
            self.rotor_speeds = actuators.thrust_to_rotor_speed(
                self.motor_thrusts,
                thrust_coefficient=self._nominal_thrust_coefficient,
            )
        self.prop_thrust_coefficients = np.full(
            4, self._nominal_thrust_coefficient, dtype=float
        )
        self.rotor_yaw_equivalent_forces = self.motor_thrusts.copy()
        self.commanded_motor_thrusts_prefault = self.motor_thrusts.copy()
        self.commanded_motor_thrusts = self.motor_thrusts.copy()
        self.rotor_fault_active = False
        # Ground-effect thrust multiplier applied in the last update() (1.0
        # when params.ground_effect_enabled is False). Read by the IMU model
        # so the accelerometer sees the same thrust the plant applies.
        self.ground_effect_factor = 1.0

    def world_frame(self):
        """position returns a 3x6 matrix
        where row is [x, y, z] column is m1 m2 m3 m4 origin h
        """
        origin = self.state[0:3]
        rot = self.rotation_body_to_world()
        wHb = np.r_[np.c_[rot, origin], np.array([[0, 0, 0, 1]])]
        quadBodyFrame = self.body_frame().T
        quadWorldFrame = wHb.dot(quadBodyFrame)
        world_frame = quadWorldFrame[0:3]
        return world_frame

    def body_frame(self):
        """Motor positions m1..m4, the origin and a point above it (6x4,
        homogeneous), for drawing.  Body frame: +x forward, +y left, +z up;
        Crazyflie numbering M1 front-right (CCW), M2 rear-right (CW), M3
        rear-left (CCW), M4 front-left (CW), as in the profile's mixer."""
        xy = self.actuator_profile.mixer_arm
        return np.array([( xy, -xy, 0, 1),
                         (-xy, -xy, 0, 1),
                         (-xy,  xy, 0, 1),
                         ( xy,  xy, 0, 1),
                         (0, 0, 0, 1),
                         (0, 0, BODY_DRAW_HEIGHT, 1)])

    def position(self):
        return self.state[0:3]

    def velocity(self):
        return self.state[3:6]

    def attitude(self):
        return self.state[6:9]

    def omega(self):
        return self.state[9:12]

    def rotor_speed_rad_s(self):
        """Mechanical rotor speeds; plant truth for sensor simulation only."""
        return self.rotor_speeds.copy()

    def rotation_world_to_body(self):
        if self._use_quaternion:
            return self.rotation_body_to_world().T
        phi, theta, psi = self.attitude()
        return RPYToRot(float(phi), float(theta), float(psi))

    def rotation_body_to_world(self):
        if self._use_quaternion:
            return quat_to_rot_body_to_world(self.quat)
        return self.rotation_world_to_body().T

    @staticmethod
    def euler_rates(phi, theta, p, q, r):
        # This matches the simulator's RPYToRot convention; it is singular when
        # cos(phi) approaches zero.
        cos_phi = np.cos(phi)
        if abs(cos_phi) < 1.0e-6:
            cos_phi = np.copysign(1.0e-6, cos_phi if cos_phi != 0.0 else 1.0)

        sin_phi = np.sin(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tan_phi = sin_phi / cos_phi

        phi_dot = p * cos_theta + r * sin_theta
        theta_dot = q + (p * sin_theta - r * cos_theta) * tan_phi
        psi_dot = (-p * sin_theta + r * cos_theta) / cos_phi
        return np.array([phi_dot, theta_dot, psi_dot], dtype=float)

    def _rotor_moment(self, omega):
        """(-omega_y h, omega_x h, reaction) of the rotor angular momentum."""
        speeds = self.rotor_speeds
        h = self.rotor_inertia_kgm2 * float(self._rotor_signs @ speeds) if self.rotor_gyro else 0.0
        react = 0.0
        if self.rotor_reaction and self._rotor_step["dt"]:
            react = (-self.rotor_inertia_kgm2
                     * float(self._rotor_signs @ (speeds - self._rotor_step["prev"]))
                     / self._rotor_step["dt"])
        return np.array([-omega[1] * h, omega[0] * h, react])

    def state_dot(self, state, t, F, M, external_force_world, wind_vel_world=None):
        if self.rotor_inertia_kgm2 > 0.0:
            M = np.asarray(M, dtype=float).reshape(3,) + self._rotor_moment(state[9:12])
        x, y, z, xdot, ydot, zdot, phi, theta, psi, p, q, r = state

        bRw = RPYToRot(phi, theta, psi)  # world to body rotation matrix
        wRb = bRw.T  # orthogonal matrix inverse = transpose
        # acceleration - Newton's second law of motion
        thrust_world = wRb.dot(np.array([0.0, 0.0, F]))
        vel_world = np.array([xdot, ydot, zdot])
        # Aerodynamic force (parasitic drag + optional rotor drag) acting on
        # the relative airflow v - v_wind. With every aero switch off and no
        # wind this is bit-identical to the historical world-frame quadratic
        # drag (see model/aero.py).
        aero_world = aero.aero_force_world(vel_world, wind_vel_world, bRw)
        accel = (
            thrust_world
            + external_force_world
            - np.array([0.0, 0.0, self.mass * params.g])
            + aero_world
        ) / self.mass
        eulerdot = self.euler_rates(phi, theta, p, q, r)

        # angular acceleration - Euler's equation of motion
        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        omega = np.array([p, q, r])
        damping_torque = self.angular_damping * omega
        yaw_drag_torque = np.array(
            [0.0, 0.0, self.yaw_drag_coefficient * r * abs(r)],
            dtype=float,
        )
        pqrdot = self.inv_inertia.dot(
            M.flatten()
            - np.cross(omega, self.inertia.dot(omega))
            - damping_torque
            - yaw_drag_torque
        )
        state_dot = np.zeros(12)
        state_dot[0] = xdot
        state_dot[1] = ydot
        state_dot[2] = zdot
        state_dot[3] = accel[0]
        state_dot[4] = accel[1]
        state_dot[5] = accel[2]
        state_dot[6] = eulerdot[0]
        state_dot[7] = eulerdot[1]
        state_dot[8] = eulerdot[2]
        state_dot[9] = pqrdot[0]
        state_dot[10] = pqrdot[1]
        state_dot[11] = pqrdot[2]

        return state_dot

    def _rigid_body_rates(self, omega, M):
        omega = np.asarray(omega, dtype=float).reshape(3,)
        if self.rotor_inertia_kgm2 > 0.0:
            M = np.asarray(M, dtype=float).reshape(3,) + self._rotor_moment(omega)
        damping_torque = self.angular_damping * omega
        yaw_drag_torque = np.array(
            [
                0.0,
                0.0,
                self.yaw_drag_coefficient
                * omega[2]
                * abs(omega[2]),
            ],
            dtype=float,
        )
        return self.inv_inertia.dot(
            np.asarray(M, dtype=float).reshape(3,)
            - np.cross(omega, self.inertia.dot(omega))
            - damping_torque
            - yaw_drag_torque
        )

    def _translational_accel(self, vel_world, rotation_body_to_world, F, external_force_world, wind_vel_world):
        thrust_world = rotation_body_to_world.dot(np.array([0.0, 0.0, F]))
        bRw = rotation_body_to_world.T
        aero_world = aero.aero_force_world(vel_world, wind_vel_world, bRw)
        return (
            thrust_world
            + external_force_world
            - np.array([0.0, 0.0, self.mass * params.g])
            + aero_world
        ) / self.mass

    def _rk4_quaternion_step(self, dt, F, M, external_force_world, wind_vel_world):
        """Integrate translation + quaternion attitude + body rates with RK4."""

        def derivatives(pos, vel, quat, omega):
            R = quat_to_rot_body_to_world(quat)
            accel = self._translational_accel(
                vel, R, F, external_force_world, wind_vel_world
            )
            q_dot = quat_derivative(quat, omega)
            omega_dot = self._rigid_body_rates(omega, M)
            return vel, accel, q_dot, omega_dot

        pos = self.state[0:3].copy()
        vel = self.state[3:6].copy()
        quat = self.quat.copy()
        omega = self.state[9:12].copy()

        k1 = derivatives(pos, vel, quat, omega)
        k2 = derivatives(
            pos + 0.5 * dt * k1[0],
            vel + 0.5 * dt * k1[1],
            quat + 0.5 * dt * k1[2],
            omega + 0.5 * dt * k1[3],
        )
        k3 = derivatives(
            pos + 0.5 * dt * k2[0],
            vel + 0.5 * dt * k2[1],
            quat + 0.5 * dt * k2[2],
            omega + 0.5 * dt * k2[3],
        )
        k4 = derivatives(
            pos + dt * k3[0],
            vel + dt * k3[1],
            quat + dt * k3[2],
            omega + dt * k3[3],
        )

        pos = pos + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        vel = vel + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        quat = quat + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        omega = omega + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])

        norm = float(np.linalg.norm(quat))
        if norm > 1.0e-16:
            quat = quat / norm
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        self.quat = quat
        phi, theta, psi = quat_to_rpy(quat)
        self.state[0:3] = pos
        self.state[3:6] = vel
        self.state[6] = phi
        self.state[7] = theta
        self.state[8] = psi
        self.state[9:12] = omega

    def update(
        self,
        dt,
        motor_pwm,
        external_force_world=None,
        sim_time=None,
        wind_velocity_world=None,
    ):
        if self.rotor_inertia_kgm2 > 0.0:
            self._rotor_step["prev"] = self.rotor_speeds.copy()
            self._rotor_step["dt"] = float(dt)
        # Optional battery-sag model (default OFF; when disabled not a single
        # float op below runs). Estimate the current draw from the PREVIOUS
        # step's realized thrust and relax the terminal voltage toward
        # OCV - I*R with time constant params.battery_sag_tau. Runs before any
        # actuator call so wrench_to_pwm/pwm_to_motor_forces see the sagged
        # voltage.
        if self._battery_sag_enabled:
            i_est = (
                float(np.sum(self.motor_thrusts))
                / float(np.sum(self._plant_command_upper))
            ) * params.battery_max_current_draw
            v_target = float(
                np.clip(
                    self._battery_ocv
                    - i_est * params.battery_internal_resistance,
                    params.battery_voltage_min,
                    params.battery_voltage_max,
                )
            )
            self._sag_voltage += (dt / params.battery_sag_tau) * (
                v_target - self._sag_voltage
            )
            self.battery_voltage = self._sag_voltage

        desired_motor_pwm = np.clip(
            np.asarray(motor_pwm, dtype=float).reshape(4,),
            params.pwm_min,
            params.pwm_max,
        )
        desired_motor_thrusts = actuators.pwm_to_motor_forces(
            desired_motor_pwm,
            self.battery_voltage,
            actuator_profile=self.actuator_profile,
        )
        desired_motor_thrusts_prefault = desired_motor_thrusts.copy()
        if external_force_world is None:
            external_force_world = np.zeros(3)
        external_force_world = np.asarray(
            external_force_world, dtype=float
        ).reshape(3,)

        if self._use_rotor_speed_model:
            # Fidelity path: exact ZOH is applied to mechanical rotor speed,
            # followed by the nonlinear f=cT*Omega^2 map.  It is deliberately
            # not claimed to be identical to the thrust-space exact-ZOH model.
            # Motor/ESC loss acts on speed while propeller loss acts on cT.
            speed_scale, coefficient_scale = self.rotor_fault.physical_scales(
                sim_time
            )
            speed_scale = np.asarray(speed_scale, dtype=float)
            coefficient_scale = np.asarray(coefficient_scale, dtype=float)
            # The fault's steady thrust effectiveness splits into the part
            # realized through the speed (speed_scale**2 under a quadratic map)
            # and the part applied to the produced thrust (coefficient_scale).
            # Writing it this way makes the target speed F^-1(eff * f_a) for
            # any map: under the quadratic it is algebraically identical to
            # scaling the healthy target speed by speed_scale, and under the
            # cubic it keeps lambda the target thrust effectiveness instead of
            # silently redefining it.  A damaged propeller keeps the healthy
            # target speed; only its produced thrust is scaled.
            speed_side_effectiveness = speed_scale ** 2
            if self.thrust_map == "graefe_cubic":
                target_speed = graefe_speed(
                    speed_side_effectiveness * desired_motor_thrusts_prefault
                )
            else:
                target_speed = actuators.thrust_to_rotor_speed(
                    desired_motor_thrusts_prefault,
                    thrust_coefficient=self._nominal_thrust_coefficient,
                ) * speed_scale
            taus = np.where(
                target_speed >= self.rotor_speeds,
                self.motor_tau_up,
                self.motor_tau_down,
            )
            decay = np.exp(-float(dt) / np.maximum(taus, 1.0e-9))
            self.rotor_speeds = decay * self.rotor_speeds + (
                1.0 - decay
            ) * target_speed
            self.rotor_speeds = np.clip(
                self.rotor_speeds,
                params.rotor_speed_min_rad_s,
                self._rotor_speed_limit,
            )
            self.prop_thrust_coefficients = (
                self._nominal_thrust_coefficient * coefficient_scale
            )
            if self.thrust_map == "graefe_cubic":
                self.motor_thrusts = (
                    coefficient_scale * graefe_thrust(self.rotor_speeds)
                )
            else:
                self.motor_thrusts = actuators.rotor_speed_to_thrust(
                    self.rotor_speeds, self.prop_thrust_coefficients
                )
            yaw_coefficient_scale = np.asarray(
                self.rotor_fault.yaw_torque_coefficient_scales(sim_time),
                dtype=float,
            )
            if self.thrust_map == "graefe_cubic":
                self.rotor_yaw_equivalent_forces = (
                    graefe_thrust(self.rotor_speeds) * yaw_coefficient_scale
                )
                desired_motor_thrusts = (
                    coefficient_scale * graefe_thrust(target_speed)
                )
            else:
                self.rotor_yaw_equivalent_forces = (
                    actuators.rotor_speed_to_thrust(
                        self.rotor_speeds,
                        thrust_coefficient=self._nominal_thrust_coefficient,
                    )
                    * yaw_coefficient_scale
                )
                desired_motor_thrusts = actuators.rotor_speed_to_thrust(
                    target_speed, self.prop_thrust_coefficients
                )
        elif self.actuator_plant_backend == THRUST_EXACT_ZOH:
            # Theorem-consistent plant: the state itself is realized thrust.
            # Net thrust effectiveness is represented in the target, then the
            # asymmetric first-order state is advanced by its exact ZOH map.
            speed_scale, coefficient_scale = self.rotor_fault.physical_scales(
                sim_time
            )
            speed_scale = np.asarray(speed_scale, dtype=float)
            coefficient_scale = np.asarray(coefficient_scale, dtype=float)
            desired_motor_thrusts_prefault = np.clip(
                desired_motor_thrusts_prefault,
                self._plant_command_lower,
                self._plant_command_upper,
            )
            desired_motor_thrusts = (
                desired_motor_thrusts_prefault
                * speed_scale
                * speed_scale
                * coefficient_scale
            )
            taus = np.where(
                desired_motor_thrusts >= self.motor_thrusts,
                self.motor_tau_up,
                self.motor_tau_down,
            )
            decay = np.exp(-float(dt) / np.maximum(taus, 1.0e-9))
            self.motor_thrusts = (
                decay * self.motor_thrusts
                + (1.0 - decay) * desired_motor_thrusts
            )
            self.motor_thrusts = np.maximum(self.motor_thrusts, 0.0)

            # Preserve the existing mechanism-dependent steady reaction-torque
            # semantics in a separate thrust-equivalent yaw state.  This does
            # not make the thrust-space plant equivalent to the RPM plant.
            yaw_coefficient_scale = np.asarray(
                self.rotor_fault.yaw_torque_coefficient_scales(sim_time),
                dtype=float,
            )
            desired_yaw_equivalent = (
                desired_motor_thrusts_prefault
                * speed_scale
                * speed_scale
                * yaw_coefficient_scale
            )
            yaw_taus = np.where(
                desired_yaw_equivalent >= self.rotor_yaw_equivalent_forces,
                self.motor_tau_up,
                self.motor_tau_down,
            )
            yaw_decay = np.exp(
                -float(dt) / np.maximum(yaw_taus, 1.0e-9)
            )
            self.rotor_yaw_equivalent_forces = (
                yaw_decay * self.rotor_yaw_equivalent_forces
                + (1.0 - yaw_decay) * desired_yaw_equivalent
            )

            # This is a nominal thrust-equivalent speed for sensor plumbing;
            # use rotor_speed_exact_zoh when physical RPM fidelity is required.
            if self.thrust_map == "graefe_cubic":
                self.rotor_speeds = graefe_speed(self.motor_thrusts)
            else:
                self.rotor_speeds = actuators.thrust_to_rotor_speed(
                    self.motor_thrusts,
                    thrust_coefficient=self._nominal_thrust_coefficient,
                )
            self.rotor_speeds = np.clip(
                self.rotor_speeds,
                params.rotor_speed_min_rad_s,
                self._rotor_speed_limit,
            )
            self.prop_thrust_coefficients[:] = self._nominal_thrust_coefficient
        else:
            desired_motor_thrusts = actuators.apply_rotor_fault(
                desired_motor_thrusts,
                self.rotor_fault,
                sim_time=sim_time,
            )
            # Legacy first-order thrust dynamics are deliberately unchanged.
            taus = np.where(
                desired_motor_thrusts >= self.motor_thrusts,
                self.motor_tau_up,
                self.motor_tau_down,
            )
            alpha = np.clip(dt / np.maximum(taus, 1e-6), 0.0, 1.0)
            self.motor_thrusts = self.motor_thrusts + alpha * (
                desired_motor_thrusts - self.motor_thrusts
            )
            self.motor_thrusts = np.maximum(self.motor_thrusts, 0.0)
            self.rotor_speeds = actuators.thrust_to_rotor_speed(
                self.motor_thrusts,
                thrust_coefficient=self._nominal_thrust_coefficient,
            )
            self.prop_thrust_coefficients[:] = self._nominal_thrust_coefficient
            self.rotor_yaw_equivalent_forces = self.motor_thrusts.copy()
        self.motor_pwm = desired_motor_pwm
        self.commanded_motor_thrusts_prefault = desired_motor_thrusts_prefault
        self.commanded_motor_thrusts = desired_motor_thrusts
        self.rotor_fault_active = self.rotor_fault.is_active(sim_time)

        # Realized force/moment after actuator dynamics. Ground effect (when
        # enabled) scales the realized thrusts aerodynamically; the factor is
        # stored so the IMU model sees the same augmented thrust.
        if params.ground_effect_enabled:
            self.ground_effect_factor = aero.ground_effect_factor(self.state[2])
        else:
            self.ground_effect_factor = 1.0
        F_realized, M_realized = actuators.motor_forces_to_wrench(
            self.motor_thrusts * self.ground_effect_factor,
            yaw_equivalent_forces=(
                self.rotor_yaw_equivalent_forces * self.ground_effect_factor
            ),
            actuator_profile=self.actuator_profile,
        )

        if self._use_quaternion:
            self._rk4_quaternion_step(
                dt,
                float(F_realized),
                M_realized.reshape(3,),
                external_force_world,
                wind_velocity_world,
            )
        else:
            self.state = integrate.odeint(
                self.state_dot,
                self.state,
                [0, dt],
                args=(
                    F_realized,
                    M_realized,
                    external_force_world,
                    wind_velocity_world,
                ),
            )[1]
            # Keep the mirror quaternion consistent for callers that inspect it.
            self.quat = rpy_to_quat(
                float(self.state[6]), float(self.state[7]), float(self.state[8])
            )

        # Ground contact model: z=0 is an impenetrable floor.
        if self.state[2] < 0.0:
            self.state[2] = 0.0
        if self.state[2] <= 0.0 and self.state[5] < 0.0:
            self.state[5] = 0.0

    def rotor_fault_state(self, sim_time=None):
        return self.rotor_fault.state_label(sim_time)
