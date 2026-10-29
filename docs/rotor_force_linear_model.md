# Rotor-Force Linearized Plant

This document starts from the hover-linearized plant that uses total thrust and
body moments as inputs, then rewrites the same linear plant with the four rotor
forces as inputs.

The important point is that this is only a change of input coordinates. The
state matrix `A` does not change. Only the input matrix changes.

## State And Inputs

The 12-state order is:

```text
x = [
  x, y, z,
  xdot, ydot, zdot,
  phi, theta, psi,
  p, q, r
]
```

The original wrench input is:

```text
u_wrench = [F, Mx, My, Mz]
```

where:

```text
F        total thrust along body +z
Mx,My,Mz body moments
```

The rotor-force input is:

```text
u_rotor = [f1, f2, f3, f4]
```

where `fi` is the thrust force from rotor `i`, in Newtons.

## Linearized Plant With Wrench Input

Strictly, the linearized model is written in perturbation variables:

```text
delta_x_dot = A * delta_x + B_wrench * delta_u_wrench
```

where:

```text
delta_x        = x - x_eq
delta_u_wrench = u_wrench - u_wrench_eq
```

For level hover:

```text
x_eq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

u_wrench_eq = [mass * g, 0, 0, 0]
            = [0.3924, 0, 0, 0]
```

If you write this as:

```text
x_dot = A*x + B*u
```

then `x` and `u` should be understood as perturbations from hover, not the
absolute state and absolute input.

## Wrench From Rotor Forces

The simulator uses the mixer geometry in `crazyflie_sim/model/actuators.py`.

Define:

```text
a = mixer_arm
k = thrust_to_torque
```

The mapping from rotor forces to total thrust and moments is:

```text
F  = f1 + f2 + f3 + f4

Mx = a * (-f1 - f2 + f3 + f4)
My = a * (-f1 + f2 + f3 - f4)
Mz = k * (-f1 + f2 - f3 + f4)
```

In matrix form:

```text
u_wrench = G * u_rotor

G =
[
  [ 1,  1,  1,  1],
  [-a, -a,  a,  a],
  [-a,  a,  a, -a],
  [-k,  k, -k,  k]
]
```

At hover, the four rotor forces are equal:

```text
f_hover = mass * g / 4
        = 0.0981 N

u_rotor_eq = [0.0981, 0.0981, 0.0981, 0.0981]
```

For perturbations:

```text
delta_u_wrench = G * delta_u_rotor
```

where:

```text
delta_u_rotor = u_rotor - u_rotor_eq
```

## Converted Linearized Plant

Substitute the rotor-force mapping into the wrench-input model:

```text
delta_x_dot =
  A * delta_x
  + B_wrench * delta_u_wrench

delta_x_dot =
  A * delta_x
  + B_wrench * G * delta_u_rotor
```

Therefore:

```text
B_rotor = B_wrench * G
```

and the rotor-force-input model is:

```text
delta_x_dot = A * delta_x + B_rotor * delta_u_rotor
```

## Symbolic Rotor-Force B Matrix

Using the hover symbolic wrench-input matrix:

```text
B_wrench =
[
  [0,   0,    0,    0],
  [0,   0,    0,    0],
  [0,   0,    0,    0],

  [0,   0,    0,    0],
  [0,   0,    0,    0],
  [1/m, 0,    0,    0],

  [0,   0,    0,    0],
  [0,   0,    0,    0],
  [0,   0,    0,    0],

  [0, 1/Ix,   0,    0],
  [0,   0, 1/Iy,    0],
  [0,   0,    0, 1/Iz]
]
```

the rotor-force input matrix becomes:

```text
B_rotor =
[
  [0,       0,       0,       0],
  [0,       0,       0,       0],
  [0,       0,       0,       0],

  [0,       0,       0,       0],
  [0,       0,       0,       0],
  [1/m,     1/m,     1/m,     1/m],

  [0,       0,       0,       0],
  [0,       0,       0,       0],
  [0,       0,       0,       0],

  [-a/Ix,  -a/Ix,   a/Ix,   a/Ix],
  [-a/Iy,   a/Iy,   a/Iy,  -a/Iy],
  [-k/Iz,   k/Iz,  -k/Iz,   k/Iz]
]
```

This makes the input meaning more physical:

```text
all four rotors increase together -> vertical acceleration
front/back or left/right differences -> roll/pitch acceleration
opposing spin-pair differences -> yaw acceleration
```

## Current Numeric Values

Current parameters:

```text
mass = 0.0400 kg
g = 9.81 m/s^2

mixer_arm = 0.03535533905932738 m
thrust_to_torque = 0.00569278844371417

Ix = 2.39e-5 kg*m^2
Iy = 2.39e-5 kg*m^2
Iz = 3.23e-5 kg*m^2
```

The numeric rotor-to-wrench matrix is:

```text
G =
[
  [ 1,           1,           1,           1],
  [-0.03535534, -0.03535534,  0.03535534,  0.03535534],
  [-0.03535534,  0.03535534,  0.03535534, -0.03535534],
  [-0.00569279,  0.00569279, -0.00569279,  0.00569279]
]
```

## Numeric A Matrix

The `A` matrix is unchanged by the input conversion:

```text
A =
[
  [0, 0, 0, 1,           0,           0, 0,     0,    0, 0,           0,           0],
  [0, 0, 0, 0,           1,           0, 0,     0,    0, 0,           0,           0],
  [0, 0, 0, 0,           0,           1, 0,     0,    0, 0,           0,           0],
  [0, 0, 0, -0.00000013, 0,           0, 0,     9.81, 0, 0,           0,           0],
  [0, 0, 0, 0,          -0.00000013,  0, -9.81, 0,    0, 0,           0,           0],
  [0, 0, 0, 0,           0,          -0.00000016, 0,   0,    0, 0,     0,           0],
  [0, 0, 0, 0,           0,           0, 0,     0,    0, 1,           0,           0],
  [0, 0, 0, 0,           0,           0, 0,     0,    0, 0,           1,           0],
  [0, 0, 0, 0,           0,           0, 0,     0,    0, 0,           0,           1],
  [0, 0, 0, 0,           0,           0, 0,     0,    0, -0.08368201, 0,           0],
  [0, 0, 0, 0,           0,           0, 0,     0,    0, 0,          -0.08368201, 0],
  [0, 0, 0, 0,           0,           0, 0,     0,    0, 0,           0,          -0.09287926]
]
```

## Numeric B Matrix With Wrench Input

This is the original `B` for:

```text
u_wrench = [F, Mx, My, Mz]
```

```text
B_wrench =
[
  [0,  0,           0,           0],
  [0,  0,           0,           0],
  [0,  0,           0,           0],
  [0,  0,           0,           0],
  [0,  0,           0,           0],
  [25, 0,           0,           0],
  [0,  0,           0,           0],
  [0,  0,           0,           0],
  [0,  0,           0,           0],
  [0,  41841.0041841, 0,         0],
  [0,  0,           41841.0041841, 0],
  [0,  0,           0,           30959.75232198]
]
```

## Numeric B Matrix With Rotor-Force Input

This is the converted matrix for:

```text
u_rotor = [f1, f2, f3, f4]
```

```text
B_rotor =
[
  [0,            0,            0,           0],
  [0,            0,            0,           0],
  [0,            0,            0,           0],
  [0,            0,            0,           0],
  [0,            0,            0,           0],
  [25,           25,           25,          25],
  [0,            0,            0,           0],
  [0,            0,            0,           0],
  [0,            0,            0,           0],
  [-1479.30288951, -1479.30288951, 1479.30288951,  1479.30288951],
  [-1479.30288951,  1479.30288951, 1479.30288951, -1479.30288951],
  [-176.24732024,    176.24732024, -176.24732024,   176.24732024]
]
```

## Interpretation

The rotor-force linear model assumes the inputs are direct changes in rotor
thrust force after motor dynamics:

```text
delta_u_rotor = [
  delta_f1,
  delta_f2,
  delta_f3,
  delta_f4
]
```

It does not include PWM conversion, battery compensation, or first-order motor
lag. If those actuator dynamics matter, they should either be modeled outside
this plant or added as extra actuator states.

For controller analysis, this rotor-force form is often more useful than the
wrench-input form because it exposes the true actuator directions and limits.
