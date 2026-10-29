"""Aerodynamic force model shared by the plant and the IMU sensor model.

Three independent, config-switchable upgrades over the historical drag model
(all OFF by default; see model/params.py):

1. Rotor-induced drag ("H-force", params.rotor_drag_enabled):
   the dominant horizontal damping on nano-quadrotors.  Airflow in the rotor
   plane tilts/rotates each rotor's thrust vector, producing a force opposing
   the in-plane relative airflow, approximately linear in velocity:

       F_rotor_body = -[dx, dy, 0] * v_rel_body

   Reference: Faessler, Franchi, Scaramuzza, "Differential Flatness of
   Quadrotor Dynamics Subject to Rotor Drag" (arXiv:1604.07797).  The
   coefficients are calibrated at hover thrust (the physical coefficient
   scales ~ sqrt(thrust); a constant is the standard simplification).

2. Body-frame anisotropic parasitic drag (params.body_frame_drag_enabled):
   the drag cross-section is attached to the airframe, so the xy/z drag
   factors act on the BODY-frame relative airflow, componentwise quadratic:

       F_drag_body = -[kx, ky, kz] * |v_rel_body| .* v_rel_body

   The legacy model instead applies the factors in the WORLD frame with the
   scalar |v| coupling all axes, which is only meaningful near hover.

3. Wind as a velocity field (model.wind.WindModel with as_velocity=True):
   all aero forces act on the relative airflow v_rel = v - v_wind, so a wind
   field enters the dynamics through drag (and the accelerometer) exactly as
   on the real vehicle instead of as an arbitrary external force.

Ground effect (params.ground_effect_enabled) is a thrust multiplier, modeled
in ground_effect_factor() and applied by Quadcopter.update.

Bit-identity invariant: with every switch OFF and wind_vel_world=None,
aero_force_world() returns exactly the negation of the historical inline
expression `drag_factors * |v| * v` (same operation order; IEEE-754
x - y == x + (-y)), so the default plant is numerically unchanged
(enforced by tests/test_golden.py).
"""

import numpy as np

import crazyflie_sim.model.params as params


def _parasitic_drag_factors():
    return np.array(
        [
            params.aero_drag_factor_xy,
            params.aero_drag_factor_xy,
            params.aero_drag_factor_z,
        ]
    )


def _rotor_drag_coefficients():
    return np.array(
        [
            params.rotor_drag_coefficient_x,
            params.rotor_drag_coefficient_y,
            0.0,  # rotor drag acts in the rotor plane only
        ]
    )


def aero_force_world(vel_world, wind_vel_world, bRw):
    """Total aerodynamic force on the quad in the world frame (N).

    The returned force is ADDED to the other forces in the translational
    dynamics (it opposes the relative airflow).

    vel_world      : vehicle ground velocity, world frame (3,)
    wind_vel_world : wind velocity field, world frame (3,), or None (calm air)
    bRw            : world-to-body rotation matrix (3, 3)
    """
    vel_world = np.asarray(vel_world, dtype=float).reshape(3)
    if wind_vel_world is None:
        v_rel = vel_world
    else:
        v_rel = vel_world - np.asarray(wind_vel_world, dtype=float).reshape(3)

    if not (params.rotor_drag_enabled or params.body_frame_drag_enabled):
        # Legacy model: world-frame quadratic with |v| coupling, bit-identical
        # to the historical inline expression in Quadcopter.state_dot.
        drag_factors = _parasitic_drag_factors()
        vel_norm = np.linalg.norm(v_rel)
        return -(drag_factors * vel_norm * v_rel)

    force_world = np.zeros(3)
    v_rel_body = None
    if params.body_frame_drag_enabled:
        v_rel_body = bRw.dot(v_rel)
        factors = _parasitic_drag_factors()
        force_world = force_world + bRw.T.dot(
            -factors * np.abs(v_rel_body) * v_rel_body
        )
    else:
        drag_factors = _parasitic_drag_factors()
        vel_norm = np.linalg.norm(v_rel)
        force_world = force_world - drag_factors * vel_norm * v_rel
    if params.rotor_drag_enabled:
        if v_rel_body is None:
            v_rel_body = bRw.dot(v_rel)
        force_world = force_world + bRw.T.dot(
            -_rotor_drag_coefficients() * v_rel_body
        )
    return force_world


def ground_effect_factor(z):
    """Thrust multiplier in ground effect, T_IGE / T_OGE >= 1.

    Cheeseman & Bennett (1955) single-rotor model:

        T_IGE / T_OGE = 1 / (1 - (R / (4z))^2)

    z is the rotor-plane height above ground (m).  z is clamped to >= R/2 so
    the factor saturates at 4/3 instead of diverging at z = R/4.  Returns
    exactly 1.0 when params.ground_effect_enabled is False.
    """
    if not params.ground_effect_enabled:
        return 1.0
    r = float(params.prop_radius)
    z_eff = max(float(z), 0.5 * r)
    ratio = r / (4.0 * z_eff)
    return 1.0 / (1.0 - ratio * ratio)
