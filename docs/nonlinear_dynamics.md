# 12-State Nonlinear Dynamics

This document explains the runtime plant model used by the simulator. This is
the model integrated during simulation. It is separate from `crazyflie_sim/model/linearization.py`,
which is only a diagnostic hover-linearization helper.

## Runtime Path

The simulation loop does not integrate the nominal `A, B` model. The runtime
path is:

```text
desired_state
-> controller
-> commanded total thrust and body moments
-> actuator mixer
-> four motor PWM commands
-> PWM-to-thrust model
-> first-order motor thrust dynamics
-> realized total thrust and body moments
-> nonlinear 12-state dynamics
```

The main code locations are:

```text
crazyflie_sim/sim/simulation.py                 simulation loop
crazyflie_sim/controllers/pid.py                 baseline cascaded PID controller
crazyflie_sim/model/actuators.py                 mixer and PWM/thrust conversion
crazyflie_sim/model/quadcopter.py                nonlinear plant dynamics
```

## State Definition

The plant state is:

```text
x_state = [
  x, y, z,
  xdot, ydot, zdot,
  phi, theta, psi,
  p, q, r
]
```

Grouped by meaning:

```text
position_world = [x, y, z]
velocity_world = [xdot, ydot, zdot]
attitude       = [phi, theta, psi]   roll, pitch, yaw
body_rates     = [p, q, r]
```

The attitude is stored directly as Euler angles. The angular rates `p, q, r`
are body-frame angular velocity components, not Euler angle derivatives.

## Plant Inputs

The nonlinear dynamics receive a realized body wrench:

```text
F = total thrust along body +z
M = [Mx, My, Mz]
```

The controller does not apply `F, Mx, My, Mz` directly to the plant. It first
passes them through the actuator model. The plant receives the realized wrench
after PWM conversion, rotor faults, and first-order motor thrust lag.

## Command To Wrench

The baseline controller computes:

```text
F = mass * (g + az_cmd)
```

Then the rate controller produces firmware-like roll, pitch, and yaw commands:

```text
u_roll
u_pitch
u_yaw
```

These are scaled into physical body moments:

```text
Mx = max_mx * u_roll  / MOTOR_MIX_CMD_MAX
My = max_my * u_pitch / MOTOR_MIX_CMD_MAX
Mz = max_mz * u_yaw   / MOTOR_MIX_CMD_MAX
```

The maximum moments come from the actuator geometry and thrust limits:

```text
max_mx = 2 * mixer_arm * maxF_per_motor
max_my = 2 * mixer_arm * maxF_per_motor
max_mz = 2 * thrust_to_torque * maxF_per_motor
```

## Wrench To Motor Forces

The actuator mixer maps the desired body wrench into per-rotor force setpoints:

```text
a = mixer_arm
k = thrust_to_torque

f1 = F/4 - Mx/(4a) - My/(4a) - Mz/(4k)
f2 = F/4 - Mx/(4a) + My/(4a) + Mz/(4k)
f3 = F/4 + Mx/(4a) + My/(4a) - Mz/(4k)
f4 = F/4 + Mx/(4a) - My/(4a) + Mz/(4k)
```

This is implemented by `mix_wrench_to_motor_forces()` in
`crazyflie_sim/model/actuators.py`.

Those force setpoints are converted into firmware-style thrust commands and
then into PWM using the Bitcraze brushless motor-voltage-to-thrust curve.

## PWM To Realized Wrench

During `Quadcopter.update()`, the PWM command is converted back into static
per-motor thrust:

```text
motor_voltage = battery_voltage * pwm / pwm_max

motor_force =
  c0
  + c1 * motor_voltage
  + c2 * motor_voltage^2
  + c3 * motor_voltage^3
```

Then rotor fault effectiveness is applied if a rotor fault is active.

After that, the simulator applies first-order motor thrust dynamics:

```text
tau = motor_tau_up    if desired_thrust >= current_thrust
tau = motor_tau_down  otherwise

alpha = dt / tau

current_thrust =
  current_thrust + alpha * (desired_thrust - current_thrust)
```

The realized motor thrusts are converted back into the actual wrench:

```text
F  = f1 + f2 + f3 + f4

Mx = a * (f3 + f4 - f1 - f2)
My = a * (f2 + f3 - f1 - f4)
Mz = k * (-f1 + f2 - f3 + f4)
```

That realized `F, Mx, My, Mz` is what drives the nonlinear plant.

## Translational Dynamics

The position dynamics are:

```text
position_dot = velocity
```

The velocity dynamics are:

```text
velocity_dot =
  (
    R_body_to_world * [0, 0, F]
    + external_force_world
    - [0, 0, mass * g]
    - drag_world
  ) / mass
```

`RPYToRot(phi, theta, psi)` returns the world-to-body rotation matrix:

```text
R_world_to_body = RPYToRot(phi, theta, psi)
R_body_to_world = transpose(R_world_to_body)
```

So body thrust is rotated into the world frame as:

```text
thrust_world = R_body_to_world * [0, 0, F]
```

The drag model is quadratic in speed:

```text
speed = norm(velocity_world)

drag_world = [
  aero_drag_factor_xy * speed * xdot,
  aero_drag_factor_xy * speed * ydot,
  aero_drag_factor_z  * speed * zdot
]
```

The code computes this drag vector as a positive vector in the direction of
velocity, then subtracts it from the force balance.

## Euler-Angle Kinematics

The simulator stores attitude as:

```text
attitude = [phi, theta, psi]
```

The angular velocity state is:

```text
omega_body = [p, q, r]
```

The Euler angle rates are computed from `p, q, r` as:

```text
phi_dot =
  p * cos(theta)
  + r * sin(theta)

theta_dot =
  q
  + (p * sin(theta) - r * cos(theta)) * tan(phi)

psi_dot =
  (-p * sin(theta) + r * cos(theta)) / cos(phi)
```

Equivalently:

```text
[phi_dot, theta_dot, psi_dot] = E(phi, theta) * [p, q, r]

E(phi, theta) =
[
  [ cos(theta),              0,  sin(theta)             ],
  [ sin(theta)*tan(phi),     1, -cos(theta)*tan(phi)    ],
  [-sin(theta)/cos(phi),     0,  cos(theta)/cos(phi)    ]
]
```

This expression matches the simulator's `RPYToRot()` convention. It is singular
when `cos(phi)` approaches zero.

## Rotational Dynamics

The angular acceleration follows rigid-body Euler equations:

```text
omega_dot =
  invI * (
    M
    - omega cross (I * omega)
    - angular_damping * omega
  )
```

where:

```text
omega = [p, q, r]
M     = [Mx, My, Mz]
I     = inertia matrix
invI  = inverse inertia matrix
```

The term `omega cross (I * omega)` is gyroscopic coupling. The damping term is
a simple linear angular damping torque:

```text
damping_torque = angular_damping * omega
```

## Full State Derivative

Putting the pieces together:

```text
x_dot      = xdot
y_dot      = ydot
z_dot      = zdot

xdot_dot   = accel_world_x
ydot_dot   = accel_world_y
zdot_dot   = accel_world_z

phi_dot    = p*cos(theta) + r*sin(theta)
theta_dot  = q + (p*sin(theta) - r*cos(theta))*tan(phi)
psi_dot    = (-p*sin(theta) + r*cos(theta)) / cos(phi)

p_dot      = omega_dot_x
q_dot      = omega_dot_y
r_dot      = omega_dot_z
```

This 12-state nonlinear model is integrated by `scipy.integrate.odeint()` in
`Quadcopter.update()`.

