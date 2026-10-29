"""A minimal working controller: copy this file into crazyflie_sim/controllers/ to fly it.

A PID loop on position gives the force the rotors must produce; a geometric
attitude loop (Lee et al., 2010) turns the body z axis toward that force,
with a softer yaw gain so a weak rotor does not spend roll/pitch authority on
yaw.  It hovers, rejects the benchmark gust and tracks the benchmark path;
improving it is left to you.
"""

import numpy as np

from crazyflie_sim.controllers.base import ControllerBase
from crazyflie_sim.controllers.registry import register
from crazyflie_sim.model import actuators, params
from crazyflie_sim.model.actuator_profile import get_actuator_profile


@register("my_ctrl", label="My controller", color="#d62728")
class MyController(ControllerBase):
    def __init__(self, dt, actuator_profile=None, kp=10.0, ki=8.0, kd=6.0, i_limit=1.0,
                 k_att=400.0, k_rate=30.0, yaw_weight=0.25):
        self.dt = dt  # every argument except dt and actuator_profile is a config gain
        self.actuator_profile = get_actuator_profile(actuator_profile)
        self.kp, self.ki, self.kd, self.i_limit = kp, ki, kd, i_limit
        self.k_att, self.k_rate = k_att, k_rate
        self.axis_weight = np.array([1.0, 1.0, yaw_weight])
        self.reset()

    def set_dt(self, dt): self.dt = dt
    def reset(self): self.i_err = np.zeros(3)
    def diagnostics(self): return {}

    def run(self, quad, des_state, sim_time=0.0):
        vehicle = self.actuator_profile
        # Position loop: the world-frame force that gives the desired acceleration.
        e = des_state.pos - quad.position()
        self.i_err = np.clip(self.i_err + e * self.dt, -self.i_limit, self.i_limit)
        acc = (des_state.acc + self.kp * e + self.ki * self.i_err
               + self.kd * (des_state.vel - quad.velocity()))
        force = vehicle.mass * (acc + np.array([0.0, 0.0, params.g]))
        R = quad.rotation_body_to_world()
        thrust = force @ R[:, 2]  # what the rotors deliver along the body z axis

        # Attitude loop: desired body axes from the force direction and the yaw reference.
        z_d = force / np.linalg.norm(force)
        heading = np.array([np.cos(des_state.yaw), np.sin(des_state.yaw), 0.0])
        y_d = np.cross(z_d, heading)
        y_d /= np.linalg.norm(y_d)
        R_d = np.column_stack((np.cross(y_d, z_d), y_d, z_d))
        E = R_d.T @ R - R.T @ R_d
        e_R = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])  # attitude error (vee map)
        moment = vehicle.inertia @ (self.axis_weight * (-self.k_att * e_R - self.k_rate * quad.omega()))

        # Mixer, rotor limits and the PWM map of the run's vehicle: per-rotor PWM.
        return actuators.wrench_to_pwm(thrust, moment, quad.battery_voltage,
                                       actuator_profile=vehicle)
