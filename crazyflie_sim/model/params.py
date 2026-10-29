"""Simulation-wide constants: physics, the Crazyflie firmware command
interface, battery, sensors, the aerodynamic model and simulator switches.

The vehicle itself (mass, inertia, geometry, rotor and motor constants, thrust
curve, rotor inertia) is not here: it is an ``ActuatorProfile`` in
model/actuator_profile.py, and the plant and every controller read the run's
profile.  The default profile, crazyflie21_physical, carries the Crazyflie 2.1
Brushless values that used to live in this file.
"""


g = 9.81  # m/s/s

# Crazyflie firmware uses 16-bit motor PWM/thrust commands internally.
pwm_min = 0.0
pwm_max = 65535.0

# Firmware per-motor thrust command range, Bitcraze cf21bl THRUST_MIN and
# THRUST_MAX (platform_defaults_cf21bl.h).  The 16-bit thrust command is
# scaled to a profile's per-rotor ceiling; these two set the idle fraction
# THRUST_MIN / THRUST_MAX of that ceiling.
thrust_cmd_min_per_motor = 0.02136263065537499  # N
thrust_cmd_max_per_motor = 0.2  # N

# Battery-voltage model for the brushless actuator path.
# By default the simulator uses a fixed loaded supply voltage; an optional
# first-order sag model (below) can drop the terminal voltage under load
# without changing the controller interface.
battery_voltage_nominal = 3.8  # V
battery_voltage_min = 3.0  # V
battery_voltage_max = 4.2  # V

# Optional first-order battery-voltage sag model (default OFF).
# When enabled, Quadcopter.update() estimates the current draw from the
# previous step's realized thrust and relaxes the terminal voltage toward
# OCV - I*R with time constant battery_sag_tau.
battery_sag_enabled = False
battery_internal_resistance = 0.08   # ohm
battery_max_current_draw = 12.0      # A at full-throttle scale
battery_sag_tau = 0.5                # s, first-order lag

# ---------------------------------------------------------------------------
# Rotor-speed actuator model.
# ---------------------------------------------------------------------------
# The legacy simulator applies a forward-Euler lag directly to thrust.  The
# physical-fidelity path instead evolves mechanical rotor speed by exact ZOH
# and then evaluates the profile's speed-to-thrust curve.  Quadcopter also
# exposes the separate theorem-consistent ``thrust_exact_zoh`` backend; the
# two exact-ZOH paths are not dynamically identical.  This flag retains its
# historical mapping to the rotor-speed path so old configurations remain
# reproducible.
rotor_speed_model_enabled = False
rotor_speed_min_rad_s = 0.0

# BMI088 full-scale range of the Crazyflie gyro.  A 40 rad/s post-loss spin
# is about 2290 deg/s, so this range saturates.  The research-plant
# navigation IMU uses sim_gyro_range_dps.
gyro_range_dps = 2000.0
sim_gyro_range_dps = 4000.0

# Generic (pre-calibration) propeller reaction-torque loss model.  A value of
# one preserves the nominal yaw torque/thrust ratio; a value below one creates
# a bounded mismatch while ensuring reaction torque vanishes at complete loss.
prop_yaw_torque_loss_exponent = 0.9

# ---------------------------------------------------------------------------
# Standard aerodynamic drag model
# F_drag = -0.5 * rho * A * Cd * |v| * v
# where |v| is the Euclidean norm of the velocity vector.  The areas and
# coefficients describe the Crazyflie envelope and are shared by all profiles.
# ---------------------------------------------------------------------------
air_density = 1.225  # kg/m^3

# Reference frontal areas (m^2).
# Crazyflie 2.1 ~100 mm diagonal; approximate horizontal/vertical projection.
drag_area_xy = 0.008   # m^2 (horizontal cross-section: body + rotors)
drag_area_z  = 0.008   # m^2 (vertical cross-section)

# Drag coefficients (dimensionless).  Typical multirotor Cd: 0.8-1.5.
drag_coefficient_xy = 1.1
drag_coefficient_z  = 1.3

# Resulting drag factors: F_drag_axis = -factor_axis * |v| * v_axis
aero_drag_factor_xy = 0.5 * air_density * drag_area_xy * drag_coefficient_xy  # ~0.0054
aero_drag_factor_z  = 0.5 * air_density * drag_area_z  * drag_coefficient_z   # ~0.0064

# ---------------------------------------------------------------------------
# Aerodynamic realism upgrades (model/aero.py) — all default OFF.
# ---------------------------------------------------------------------------
# Propeller radius: 55 mm CF 2.1 Brushless propellers. Used by the
# ground-effect model and by the rotor-drag coefficient estimate below.
prop_radius = 0.0275  # m

# Rotor-induced drag ("H-force"): F_body = -[dx, dy, 0] * v_rel_body, opposing
# airflow in the rotor plane. This is the dominant horizontal damping on
# nano-quadrotors (~15-20x the parasitic drag at 1 m/s) and is approximately
# LINEAR in velocity, unlike the quadratic parasitic drag.
# Momentum-theory estimate at hover (per rotor T = m*g/4 = 0.098 N,
# A = pi*R^2 = 2.38e-3 m^2, induced velocity v_i = sqrt(T/(2*rho*A)) ~ 4.1 m/s):
#   d = 4 * T / v_i ~ 0.0956 N*s/m
# Calibrated at hover thrust (the physical coefficient scales ~sqrt(thrust));
# identify from hardware data when possible.
rotor_drag_enabled = False
rotor_drag_coefficient_x = 0.0956  # N/(m/s)
rotor_drag_coefficient_y = 0.0956  # N/(m/s)

# Body-frame anisotropic parasitic drag: apply aero_drag_factor_{xy,z} to the
# BODY-frame relative airflow componentwise (-k*|v_i|*v_i). When False the
# legacy world-frame model (-factor*|v|*v per world axis) is used, which is
# only meaningful near hover (the cross-section is attached to the airframe).
body_frame_drag_enabled = False

# Ground effect: multiply realized motor thrusts by
# ground_effect_factor(z) = 1/(1 - (R/4z))^2 (Cheeseman & Bennett), saturated
# at 4/3 for z < R/2. Matters for takeoff/landing and low hover.
ground_effect_enabled = False

# Optional quaternion attitude integration (default OFF => historical Euler
# odeint path, golden-traj bit-identical). When True, attitude kinematics use
# a unit quaternion (no Euler singularity near roll = ±90 deg); state[6:9]
# still mirrors RPY for the public attitude() API and logging.
use_quaternion_attitude = False
