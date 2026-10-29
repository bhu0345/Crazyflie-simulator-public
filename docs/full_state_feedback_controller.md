# Full-State Feedback (LQR) Controller

This document explains the current implementation of crazyflie_sim/controllers/full_state_feedback.py in the simulator: the model it uses, the LQR gain, reference-state generation, integral augmentation, saturation handling and the final PWM output path.

Related code:

- [Controller](../crazyflie_sim/controllers/full_state_feedback.py)
- [Hover linearization model](../crazyflie_sim/model/linearization.py)
- [Quadrotor nonlinear dynamics](../crazyflie_sim/model/quadcopter.py)
- [Moment-to-motor-PWM allocation](../crazyflie_sim/model/actuators.py)
- [Simulation entry point](../crazyflie_sim/runsim.py)
- [Unit tests](../tests/test_full_state_feedback.py)

## 1. Summary

This is a continuous-time LQR full-state feedback controller designed around the hover point. Position, velocity, Euler attitude and body angular rates form a 12-dimensional state vector, augmented with 3 position-error integral states; a single 4 × 15 feedback gain matrix computes the total thrust and the three body moments together:

~~~text
u = u_ref - K [e; e_int]
~~~

where:

- u = [F, Mx, My, Mz] is the physical control input;
- e = x - x_ref is the 12-dimensional state error;
- e_int is the position-error integral;
- u_ref is the thrust feedforward generated from the desired acceleration;
- K is obtained at controller initialization by solving the continuous-time algebraic Riccati equation (CARE).

The linear model is not used to simulate the vehicle. The LQR only computes the control command; the simulation itself still uses the nonlinear quadrotor model with aerodynamic drag, motor lag and optional rotor faults.

## 2. Place in the simulator

It is registered as controller mode `full_state_feedback`. Select it with `controller_mode: full_state_feedback` in the config, or run `python runsim.py hover_wind_default.yaml --controller full_state_feedback`. `crazyflie_sim/configs/trajectory_default.yaml`, which includes heading changes, uses Mellinger by default.

The control call chain is:

~~~mermaid
flowchart LR
    A["DesiredState: pos vel acc yaw yawdot"]
    B["Reference builder: x_ref and u_ref"]
    C["State measurement: quad state"]
    D["State error: e = x - x_ref"]
    E["Position integral: e_int"]
    F["LQR: u = u_ref - K e_aug"]
    G["Force and moment saturation"]
    H["wrench_to_pwm: four motor PWM"]
    I["Quadcopter update: motor lag + nonlinear dynamics"]
    A --> B
    C --> D
    B --> D
    D --> E
    D --> F
    E --> F
    F --> G --> H --> I
    I --> C
~~~

In each simulation control period, crazyflie_sim.sim.simulation.Simulation.step() takes the DesiredState output by the trajectory generator and calls:

~~~python
motor_pwm = control_backend.run(quad, desired_state, sim_time=t)
quad.update(dt, motor_pwm, external_force_world=..., sim_time=t)
~~~

The default control rate is 500 Hz, so the default dt = 1 / 500 s = 0.002 s.

## 3. State and input definitions

### 3.1 12-dimensional state

The state order is defined jointly by model.linearization.STATE_ORDER and the internal state of Quadcopter:

| Index | Variable | Meaning | Unit |
|---:|---|---|---|
| 0–2 | x, y, z | World-frame position | m |
| 3–5 | xdot, ydot, zdot | World-frame linear velocity | m/s |
| 6–8 | phi, theta, psi | Roll, pitch, yaw Euler angles | rad |
| 9–11 | p, q, r | Body-frame angular rates | rad/s |

The controller assembles the current state in this order:

~~~text
x = [x, y, z, xdot, ydot, zdot, phi, theta, psi, p, q, r]
~~~

### 3.2 4-dimensional control input

The LQR input is not PWM but the physical wrench:

~~~text
u = [F, Mx, My, Mz]
~~~

| Variable | Meaning | Unit |
|---|---|---|
| F | Total thrust of the four rotors, along body +z | N |
| Mx | Roll moment | N·m |
| My | Pitch moment | N·m |
| Mz | Yaw moment | N·m |

At the hover equilibrium, the nominal input is:

~~~text
u_eq = [m g, 0, 0, 0]
~~~

With the current default parameters m = 0.040 kg and g = 9.81 m/s², the hover total thrust is about 0.3924 N.

## 4. Design model: hover linearization

model.linearization.linearize_hover() takes the hover state with zero attitude, zero velocity and no external force as the operating point and computes the Jacobian of the nonlinear dynamics by central differences:

~~~text
delta_x_dot = A delta_x + B delta_u
~~~

The matrix dimensions are:

~~~text
A: 12 × 12
B: 12 × 4
~~~

The input order is [F, Mx, My, Mz], so B maps the physical thrust and body moments directly; the linear model needs no PWM or rotor-thrust allocation variables.

### 4.1 Main physical relations of the linearization

After removing the small numerical artifacts that finite differencing of the quadratic drag produces at hover, the main couplings of the model can be summarized as:

~~~text
xdot_dot ≈  g * theta
ydot_dot ≈ -g * phi
zdot_dot ≈  delta_F / m

phi_dot   ≈ p
theta_dot ≈ q
psi_dot   ≈ r

p_dot ≈ delta_Mx / Ix - rotational_damping
q_dot ≈ delta_My / Iy - rotational_damping
r_dot ≈ delta_Mz / Iz - rotational_damping
~~~

In the implementation, _sanitize_hover_A() zeroes diagonal entries smaller than 1e-5 in the velocity directions. These entries are numerical residuals of finite differencing the |v|v quadratic drag model; the genuine angular-rate damping terms are not removed.

### 4.2 Why a hover linearization

The hover linearization lets a single fixed K cover the translational, attitude and angular-rate channels. The design is simple and suits the simulator's hover/trajectory baselines. It is, however, a local model: with large attitudes, high speeds, frequent output saturation, or actuators that deviate markedly from the nominal model, the gap between the actual system and the design model grows.

## 5. Position-integral augmentation

use_integral=True by default. The controller integrates the position error into 3 additional states:

~~~text
e_int_dot = position - position_ref
~~~

The augmented system used for the LQR design is therefore of order 15:

~~~text
A_aug = [ A       0 ]
        [ I3_pos  0 ]

B_aug = [ B ]
        [ 0 ]
~~~

where I3_pos acts only on the three position components of the error. The corresponding augmented error is:

~~~text
e_aug = [e; e_int]
~~~

The position integral removes steady-state position offsets caused by persistent wind, model error or other approximately constant disturbances. With use_integral=False, the controller reduces to a plain 12-state full-state LQR, and Q must be 12 × 12.

## 6. How the LQR gain is computed

At initialization, the controller calls:

~~~python
P = solve_continuous_are(A_aug, B_aug, Q, R)
K = solve(R, B_aug.T @ P)
~~~

This gives the continuous-time LQR control law:

~~~text
u = u_ref - K e_aug
~~~

Note that this solves the continuous-time CARE, not the discrete-time DARE. dt does not enter the computation of K; it is used only for the run-time position-integral update. The controller therefore assumes a sufficiently fast control period; the default 500 Hz is consistent with its hover linear model.

### 6.1 Default Q: state weights

The default state weights use a Bryson's-rule-style normalization: each state is weighted by the inverse square of its allowed deviation.

| State group | Default allowed deviation |
|---|---:|
| Position x,y,z | 0.50 m |
| Velocity xdot,ydot,zdot | 0.80 m/s |
| Attitude phi,theta | 0.35 rad |
| Yaw psi | 0.50 rad |
| Angular rate p,q | 4.00 rad/s |
| Yaw rate r | 5.00 rad/s |
| Position integral e_int | Separate weight of 512 |

The diagonal of the default Q is:

~~~text
[4, 4, 4,
 1.5625, 1.5625, 1.5625,
 8.1633, 8.1633, 4,
 0.0625, 0.0625, 0.04,
 512, 512, 512]
~~~

### 6.2 Default R: input weights

The default input scales are:

~~~text
delta_F  = 0.40 N
delta_Mx = 0.014 N·m
delta_My = 0.014 N·m
delta_Mz = 0.0023 N·m
~~~

R uses the inverse squares of these scales:

~~~text
R = diag(1 / delta_F²,
         1 / delta_Mx²,
         1 / delta_My²,
         1 / delta_Mz²)
~~~

Because the Crazyflie's yaw-moment authority is small, the input penalty on Mz is naturally larger than on the roll/pitch moments.

### 6.3 Sanity check of the current default gain

After initialization, the default K is 4 × 15. For the augmented closed-loop matrix computed with the current default parameters,

~~~text
A_cl = A_aug - B_aug K
~~~

the largest real part of its eigenvalues is approximately:

~~~text
max(real(eigenvalues(A_cl))) = -2.38399
~~~

The closed loop of the linear, unsaturated design model is therefore asymptotically stable. This is not equivalent to a proof of global stability with saturation, motor lag and nonlinear dynamics.

## 7. Reference state and feedforward

The controller's des_state input comes from crazyflie_sim.controllers.base.DesiredState and contains:

~~~text
pos, vel, acc, yaw, yawdot
~~~

_reference() converts it into x_ref and u_ref.

### 7.1 Reference state

~~~text
x_ref = [pos_d,
         vel_d,
         phi_ref, theta_ref, yaw_d,
         0, 0, yawdot_d]
~~~

The roll/pitch references are computed from the desired horizontal acceleration and the desired yaw:

~~~text
phi_ref   = (acc_x sin(yaw_d) - acc_y cos(yaw_d)) / g
theta_ref = (acc_x cos(yaw_d) + acc_y sin(yaw_d)) / g
~~~

phi_ref and theta_ref are each limited to ±max_tilt, with default max_tilt = 50°. The roll/pitch rate references are zero, and the yaw rate reference is yawdot_d.

### 7.2 Thrust feedforward

The vertical acceleration is converted directly into the total-thrust reference:

~~~text
u_ref = [m (g + acc_z), 0, 0, 0]
~~~

The thrust is then clipped to the simulator's total-thrust range [minF, maxF]. For a hover command acc = [0,0,0]:

~~~text
x_ref = [desired_position, 0,0,0, 0,0,desired_yaw, 0,0,0]
u_ref = [m g, 0, 0, 0]
~~~

The state error is defined as the actual state minus the reference state:

~~~text
e = x - x_ref
~~~

The yaw error is wrapped to [-pi, pi] by _wrap_pi(), which avoids a spurious large angle error when crossing ±pi.

## 8. Execution in each control period

run_wrench() performs these steps:

1. Read the quadrotor's current 12-dimensional state.
2. Build x_ref and u_ref from des_state.
3. Compute e = x - x_ref, handling yaw wrapping.
4. Compute a candidate position integral:

   ~~~text
   candidate_int = clip(e_int + position_error * dt,
                         -int_limit, int_limit)
   ~~~

5. Compute the unsaturated wrench with the candidate integral state:

   ~~~text
   u_unsat = u_ref - K [e; candidate_int]
   ~~~

6. Clip the total thrust and each of the three moments.
7. Accept the candidate integral state only if the wrench is not saturated in this period; if it is saturated, keep the previous e_int. This is a conditional-integration anti-windup scheme.
8. Store the diagnostics and return F and M with shape (3,1).

The default limits are:

~~~text
F ∈ [0, actuator_profile.max_total_thrust]

|Mx| <= 2 * mixer_arm * maxF_per_motor
|My| <= 2 * mixer_arm * maxF_per_motor
|Mz| <= 2 * thrust_to_torque * maxF_per_motor
~~~

With the current parameters, the three moment limits are approximately:

~~~text
[0.01414, 0.01414, 0.00228] N·m
~~~

## 9. From wrench to PWM to nonlinear dynamics

After run_wrench(), run() calls:

~~~python
motor_pwm = actuators.wrench_to_pwm(
    total_thrust=force_cmd,
    moment=moment,
    battery_voltage=quad.battery_voltage,
)
~~~

The actuator path is:

~~~text
[F, Mx, My, Mz]
    ↓
quadrotor mixer
    ↓
target thrust of each rotor
    ↓
Crazyflie thrust-command mapping
    ↓
inversion of the cubic thrust curve at the battery voltage
    ↓
4 PWM channels, range [0, 65535]
~~~

After Quadcopter.update() receives the controller's PWM output, it further applies:

1. conversion from PWM to static rotor thrust;
2. rotor fault effectiveness (if enabled);
3. first-order motor thrust lag; the rise and fall time constants are both 0.050 s by default and are set by the actuator profile;
4. recombination of the motor thrusts into the realized total thrust and moments;
5. nonlinear rigid-body dynamics with gravity, aerodynamic drag and angular-rate damping;
6. time integration with scipy.integrate.odeint.

The LQR is therefore designed with the ideal physical wrench as its input, while the controller's output interface remains the four PWM channels that the simulator requires.

## 10. Comparison with cascaded PID

| Aspect | Full-state feedback | Cascaded PID |
|---|---|---|
| Control structure | A single LQR gain matrix provides all feedback | Multi-level loops: position → velocity → acceleration → attitude → angular rate |
| Design model | Hover-linearized A,B | Multiple hand-tuned / firmware-style PID gains |
| States used | The 12-dimensional state directly | Each loop uses errors and intermediate setpoints, level by level |
| Steady-state error | Position-integral augmentation by default | Removed by the integrators of the PID loops |
| Input coordinates | Physical total thrust and body moments | Internal PID outputs mapped to physical moments |
| Trajectory feedforward | acc generates attitude and thrust feedforward | Also uses acceleration-to-attitude/thrust feedforward |
| Limitations | Relies on the hover linear model; more strongly local | Intuitive gain structure, but the channels are tuned as a cascade |

The controller neither replaces the simulator's nonlinear dynamics nor bypasses the PWM/motor model; it replaces only the layer that computes the wrench from the state error.

## 11. Parameters and usage

### 11.1 Enabling it in the simulator

Set `controller_mode: full_state_feedback` in the config, or specify it on the command line:

~~~bash
python runsim.py hover_wind_default.yaml --controller full_state_feedback
~~~

It can also be instantiated directly:

~~~python
import numpy as np
from crazyflie_sim.controllers.base import DesiredState
from crazyflie_sim.controllers.full_state_feedback import FullStateFeedbackController
from crazyflie_sim.model.quadcopter import Quadcopter

quad = Quadcopter((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
des_state = DesiredState(np.array([0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3), 0.0, 0.0)
backend = FullStateFeedbackController(dt=1.0 / 500.0)
motor_pwm = backend.run(quad, des_state, sim_time=0.0)
~~~

### 11.2 Constructor parameters

~~~python
FullStateFeedbackController(
    dt=1.0 / 500.0,
    mass=None,
    Q=None,
    R=None,
    use_integral=True,
    int_limit=None,
    max_tilt=None,
)
~~~

| Parameter | Default | Purpose |
|---|---|---|
| dt | 0.002 s | Position-integral update period |
| mass | actuator_profile.mass | Mass used for the thrust feedforward and the controller's equilibrium thrust |
| Q | Built-in diagonal weights | Penalizes state error |
| R | Built-in diagonal weights | Penalizes control input |
| use_integral | True | Whether to use the 3-dimensional position-integral augmentation |
| int_limit | [4,4,4] | Limit on the position-integral states |
| max_tilt | 50° | Limit on the roll/pitch obtained from the desired acceleration |

### 11.3 Tuning guidelines

- Increasing the Q weight of a group of states puts more emphasis on that state error and usually makes the control more aggressive.
- Increasing the R weight of an input makes the controller less willing to use that input; the control usually becomes more conservative.
- Increasing the integral weight or int_limit removes offsets from persistent disturbances faster, but makes large control actions more likely under strong disturbances or saturation.
- Decreasing max_tilt limits horizontal maneuvers and the attitude reference; this suits cases where lower tilt angles are wanted.
- When changing dt, keep it equal to the actual simulation/control call period, because the position integral uses this value directly.

## 12. Diagnostics

diagnostics() returns:

~~~text
force_cmd     total thrust after saturation
moment_cmd    moments after saturation
moment_si     SI copy of the moments, identical to moment_cmd
motor_pwm     most recent four-channel PWM output
e             12-dimensional state error
e_int         3-dimensional position-integral error
aug_error     augmented error actually fed to the LQR
x_ref         current reference state
u_ref         current reference wrench
u_unsat       wrench before limiting
K             LQR gain matrix
A_aug         augmented design matrix
B_aug         augmented input matrix
saturated     whether thrust or moment saturated in the current period
status_label  "||e||"
status_value  2-norm of the 12-dimensional state error
~~~

The visualization status bar uses status_label and status_value to show the controller's current error magnitude.

## 13. Verification performed

The current test file contains 7 checks, covering:

- at hover, the output is a thrust of approximately m g and near-zero moments;
- a positive x position error produces a pitch moment in the correct direction;
- a yaw error produces a nonzero yaw moment;
- reset() clears the position integral;
- the public run() returns 4 PWM values within the valid range;
- the augmented LQR closed-loop poles lie in the left half-plane;
- a 6-second hover-wind simulation stays bounded.

Result of this check:

~~~text
Ran 7 tests ... OK
~~~

The largest real part of the eigenvalues of the linear augmented closed loop is approximately -2.38399. The hover-wind test also meets both thresholds set in the test: maximum position error after disturbance onset below 0.25 m, and position-error RMS below 0.20 m.

To rerun:

~~~bash
python -m unittest tests.test_full_state_feedback -v
~~~

## 14. Limitations and caveats of the current implementation

1. **Local linear model**: A and B are computed only at the windless hover point. With large attitudes, high speeds, aggressive maneuvers and frequent saturation, the predictions of the linear model degrade.
2. **Actuators not included in the LQR design**: The LQR takes the physical wrench directly as its input; the design model does not explicitly include PWM, the cubic thrust curve or the first-order motor lag. These effects appear in the simulator plant downstream of the controller.
3. **External wind not included in the design model**: Wind is injected as an external world-frame force in Quadcopter.update(). The position integral can compensate part of a persistent offset, but it does not turn the controller into a disturbance observer.
4. **Saturation changes the closed-loop properties**: The left-half-plane result in the tests applies to the unsaturated linear model; once thrust or moment saturation occurs, the actual system is a constrained nonlinear system.
5. **Rotor faults are not a dedicated mode of this controller**: full_state_feedback still produces its output through the nominal quadrotor actuator allocation. None of the three controllers built into the public release can sustain flight after a complete single-rotor failure; fault-tolerant algorithms must be registered by the user through crazyflie_sim.controllers.registry.
6. **Implementation note on a custom mass parameter**: A/B are computed by model.linearization.linearize_hover(actuator_profile=...) from the actuator profile the controller receives (the same vehicle as the plant). The constructor's mass affects only u_eq and the reference thrust u_ref; to use a mass different from the profile, switch to another profile, or redesign the linearized model and the LQR gain consistently, rather than passing mass alone.

## 15. Where to start reading the code

Reading in the following order is the easiest way to understand the whole implementation:

1. FullStateFeedbackController.__init__(): linearization, augmentation, Q/R and K;
2. _current_state(): state concatenation order;
3. _reference(): trajectory reference to attitude/thrust feedforward;
4. _state_error(): error and yaw wrapping;
5. run_wrench(): integration, LQR, saturation and anti-windup;
6. run(): wrench to PWM;
7. Quadcopter.update(): PWM to actual rotor thrust, then to nonlinear dynamics.
