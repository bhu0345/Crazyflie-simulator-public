"""Matplotlib backend: used by ``runsim.py --record`` to write a video."""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
from crazyflie_sim.utils.utils import RotToRPY

history = deque()
history_time = deque()
time_history = np.zeros(500)
wind_force_history = np.zeros((500, 3))
yaw_rate_history = np.zeros(500)
pos_err_history = np.zeros(500)
motor_thrust_history = np.zeros((500, 4))
eff_est_history = np.full((500, 4), np.nan)
eff_truth_history = np.full((500, 4), np.nan)

force_count = 0
status_text = None
controller_status_text = None
position_error_max = np.nan
trajectory_history_window_s = None

# Mode-adaptive layout switches (from view_config; defaults = legacy layout).
view_sim_mode = ""
show_wind_panels = True
wind_as_velocity = False
wind_formula_value = ""
rotor_plot_kind = ""
event_time_s = None
event_label = ""

ROTOR_COLORS = ("#e8590c", "#1971c2", "#2f9e44", "#7048e8")

ax_3d = None
ax_force = None
ax_yaw = None
ax_err = None
ax_rotor = None
body_lines = []
waypoint_line = None
history_line = None
force_line_x = None
force_line_y = None
force_line_z = None
yaw_rate_line = None
err_line = None
rotor_lines = []
rotor_truth_lines = []

def _expand_3d_limits_to_include(point, margin=0.05):
    """Only expand current 3D limits so the provided point stays in view."""
    if ax_3d is None:
        return

    x, y, z = np.asarray(point, dtype=float).reshape(3,)

    x0, x1 = ax_3d.get_xlim3d()
    y0, y1 = ax_3d.get_ylim3d()
    z0, z1 = ax_3d.get_zlim3d()

    changed = False
    if x < x0:
        x0 = x - margin
        changed = True
    elif x > x1:
        x1 = x + margin
        changed = True

    if y < y0:
        y0 = y - margin
        changed = True
    elif y > y1:
        y1 = y + margin
        changed = True

    if z < z0:
        z0 = z - margin
        changed = True
    elif z > z1:
        z1 = z + margin
        changed = True

    if changed:
        ax_3d.set_xlim(x0, x1)
        ax_3d.set_ylim(y0, y1)
        ax_3d.set_zlim(z0, z1)

def _save_animation(anim, path, fps):
    """Write the animation to ``path`` (.mp4 needs ffmpeg; .gif uses pillow)."""
    path = str(path)
    dpi = 80
    if path.lower().endswith(".mp4"):
        if animation.writers.is_available("ffmpeg"):
            anim.save(path, dpi=dpi, writer=animation.FFMpegWriter(fps=fps, codec="libx264"))
            print("saved", path)
            return
        path = path[:-4] + ".gif"
        print("ffmpeg unavailable; writing", path)
    if animation.writers.is_available("imagemagick"):
        anim.save(path, dpi=dpi, writer="imagemagick", fps=fps)
    elif animation.writers.is_available("pillow"):
        anim.save(path, dpi=dpi, writer=animation.PillowWriter(fps=fps))
    else:
        raise RuntimeError("No GIF writer available. Install ffmpeg, imagemagick, or pillow.")
    print("saved", path)


def plot_quad_3d(waypoints, get_world_frame, frames=400, history_window_s=None,
                 view_config=None, save_path=None, fps=50):
    """
    get_world_frame is a function which return the "next" world frame to be drawn
    """
    global history, history_time, time_history, wind_force_history, yaw_rate_history
    global pos_err_history, motor_thrust_history, eff_est_history, eff_truth_history
    global force_count, status_text, controller_status_text, position_error_max
    global ax_3d, ax_force, ax_yaw, ax_err, ax_rotor, body_lines, waypoint_line, history_line
    global force_line_x, force_line_y, force_line_z, yaw_rate_line, err_line
    global rotor_lines, rotor_truth_lines
    global trajectory_history_window_s
    global view_sim_mode, show_wind_panels, wind_as_velocity, wind_formula_value
    global rotor_plot_kind, event_time_s, event_label

    cfg = dict(view_config or {})
    view_sim_mode = str(cfg.get("sim_mode", "") or "")
    # No view_config (legacy callers) keeps the historical all-panels layout.
    show_wind_panels = view_sim_mode in ("", "hover_wind")
    wind_as_velocity = bool(cfg.get("wind_as_velocity", False))
    wind_formula_value = str(cfg.get("wind_formula", "") or "")
    rotor_plot_kind = (
        str(cfg.get("rotor_plot", "thrust"))
        if view_sim_mode == "hover_rotor_fault"
        else ""
    )
    _event = cfg.get("event_time_s")
    event_time_s = None if _event is None else float(_event)
    event_label = str(cfg.get("event_label", "") or "")

    n_hist = max(2, int(frames) + 1)
    history = deque()
    history_time = deque()
    time_history = np.zeros(n_hist)
    wind_force_history = np.zeros((n_hist, 3))
    yaw_rate_history = np.zeros(n_hist)
    pos_err_history = np.zeros(n_hist)
    motor_thrust_history = np.zeros((n_hist, 4))
    eff_est_history = np.full((n_hist, 4), np.nan)
    eff_truth_history = np.full((n_hist, 4), np.nan)
    force_count = 0
    status_text = None
    controller_status_text = None
    position_error_max = np.nan
    trajectory_history_window_s = history_window_s

    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.4])
    ax_3d = fig.add_subplot(grid[0, 0], projection='3d')
    n_right = 2 + int(show_wind_panels) + int(bool(rotor_plot_kind))
    right_grid = grid[0, 1].subgridspec(n_right, 1, hspace=0.55)
    ax_err = fig.add_subplot(right_grid[0, 0])
    row = 1
    if show_wind_panels:
        ax_force = fig.add_subplot(right_grid[row, 0])
        row += 1
    else:
        ax_force = None
    if rotor_plot_kind:
        ax_rotor = fig.add_subplot(right_grid[row, 0])
        row += 1
    else:
        ax_rotor = None
    ax_yaw = fig.add_subplot(right_grid[row, 0])

    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")

    body_lines = [
        ax_3d.plot([], [], [], '-', c='cyan')[0],
        ax_3d.plot([], [], [], '-', c='red')[0],
        ax_3d.plot([], [], [], '-', c='blue', marker='o', markevery=2)[0],
    ]
    waypoint_line = ax_3d.plot([], [], [], '.', c='red', markersize=4)[0]
    history_line = ax_3d.plot([], [], [], '.', c='blue', markersize=2)[0]

    controller_status_text = ax_3d.text2D(
        0.02,
        0.99,
        "",
        transform=ax_3d.transAxes,
        va='top',
        fontweight='bold',
        fontsize=10,
        color='black',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )
    status_text = ax_3d.text2D(
        0.02,
        0.94,
        "",
        transform=ax_3d.transAxes,
        va='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )

    err_line = ax_err.plot([], [], '-', lw=1.6, color='tab:blue', label='|e|')[0]
    _style_axis(ax_err, "Position Error", ncol=1)
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("|pos err| (m)")
    ax_err.grid(True, alpha=0.3)
    ax_err.set_xlim(0.0, 10.0)
    ax_err.set_ylim(0.0, 0.05)

    if ax_force is not None:
        force_line_x = ax_force.plot([], [], '-', lw=1.6, color='tab:orange', label='Fx')[0]
        force_line_y = ax_force.plot([], [], '-', lw=1.6, color='tab:green', label='Fy')[0]
        force_line_z = ax_force.plot([], [], '-', lw=1.6, color='tab:purple', label='Fz')[0]
        _style_axis(
            ax_force,
            "Wind Velocity" if wind_as_velocity else "Wind Force",
            ncol=3,
        )
        ax_force.set_xlabel("Time (s)")
        ax_force.set_ylabel("Velocity (m/s)" if wind_as_velocity else "Force (N)")
        if wind_formula_value:
            ax_force.text(
                0.02,
                0.98,
                wind_formula_value,
                transform=ax_force.transAxes,
                va='top',
                fontsize=6.5,
                color='0.35',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
            )
        ax_force.grid(True, alpha=0.3)
        ax_force.set_xlim(0.0, 10.0)
        ax_force.set_ylim(-0.05, 0.05)
    else:
        force_line_x = force_line_y = force_line_z = None

    if ax_rotor is not None:
        rotor_lines = [
            ax_rotor.plot([], [], '-', lw=1.5, color=ROTOR_COLORS[i], label='m{}'.format(i + 1))[0]
            for i in range(4)
        ]
        if rotor_plot_kind == "effectiveness":
            rotor_truth_lines = [
                ax_rotor.plot([], [], '--', lw=1.1, color=ROTOR_COLORS[i])[0]
                for i in range(4)
            ]
            _style_axis(ax_rotor, "Rotor Effectiveness", ncol=4)
            ax_rotor.set_ylabel("effectiveness")
            ax_rotor.set_ylim(-0.05, 1.05)
        else:
            rotor_truth_lines = []
            _style_axis(ax_rotor, "Rotor Thrust", ncol=4)
            ax_rotor.set_ylabel("Thrust (N)")
            ax_rotor.set_ylim(0.0, 0.05)
        ax_rotor.set_xlabel("Time (s)")
        ax_rotor.grid(True, alpha=0.3)
        ax_rotor.set_xlim(0.0, 10.0)
    else:
        rotor_lines = []
        rotor_truth_lines = []

    yaw_rate_line = ax_yaw.plot([], [], '-', lw=1.8, color='tab:red', label=r'$\dot{\psi}$')[0]
    _style_axis(ax_yaw, "Yaw Rate", ncol=1)
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.set_ylabel(r'$\dot{\psi}$ (deg/s)')
    ax_yaw.grid(True, alpha=0.3)
    ax_yaw.set_xlim(0.0, 10.0)
    ax_yaw.set_ylim(-10.0, 10.0)

    if event_time_s is not None and np.isfinite(event_time_s):
        for ax in (ax_err, ax_force, ax_rotor, ax_yaw):
            if ax is None:
                continue
            ax.axvline(event_time_s, color='gray', ls='--', lw=1.0, alpha=0.7)
        if event_label:
            ax_err.annotate(
                event_label,
                xy=(event_time_s, 1.0),
                xycoords=('data', 'axes fraction'),
                xytext=(3, -3),
                textcoords='offset points',
                fontsize=8,
                color='gray',
                ha='left',
                va='top',
            )

    set_limit((-0.5,0.5), (-0.5,0.5), (0,3), ax=ax_3d)
    plot_waypoints(waypoints)
    an = animation.FuncAnimation(fig,
                                 anim_callback,
                                 fargs=(get_world_frame,),
                                 # Initialization must not advance the simulation.
                                 init_func=lambda: (),
                                 frames=frames, interval=1000.0 / fps, blit=False)

    if save_path:
        _save_animation(an, save_path, fps)
    else:
        plt.show()

def plot_waypoints(waypoints):
    global waypoint_line
    if waypoint_line is None:
        return
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[0] == 0:
        return
    if waypoints.shape[1] > 3:
        waypoints = waypoints[:, :3]
    waypoint_line.set_data(waypoints[:,0], waypoints[:,1])
    waypoint_line.set_3d_properties(waypoints[:,2])

def set_limit(x, y, z, ax=None):
    if ax is None:
        ax = ax_3d
    if ax is None:
        return
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.set_zlim(z)

def anim_callback(i, get_world_frame):
    payload = get_world_frame(i)
    set_frame(payload)

def _rotor_vector_or_none(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(4,)


def _style_axis(ax, title, ncol):
    """Title left-aligned; legend as a frameless horizontal strip just above
    the axes on the right, so it never occludes the curves."""
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(
        ncol=ncol,
        frameon=False,
        fontsize=7,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.98),
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.5,
        borderaxespad=0.0,
    )


def _parse_payload(payload):
    if isinstance(payload, dict):
        frame = np.asarray(payload.get("frame"))
        sim_time = float(payload.get("time", np.nan))
        wind_force = np.asarray(payload.get("wind_force_world", np.zeros(3)), dtype=float).reshape(3,)
        yaw_rate_dps = float(payload.get("yaw_rate_dps", np.nan))
        position_error_norm = float(payload.get("position_error_norm", np.nan))
        position_error_norm_for_max = float(payload.get("position_error_norm_for_max", position_error_norm))
        position_error_max_payload = float(payload.get("position_error_max", np.nan))
        waiting_for_wind = bool(payload.get("waiting_for_wind", False))
        position_error_waiting_label = str(payload.get("position_error_waiting_label", ""))
        controller_mode = str(payload.get("controller_mode", "n/a"))
        rotor_fault_case = str(payload.get("rotor_fault_case", ""))
        rotor_fault_rotor = int(payload.get("rotor_fault_rotor", 0))
        rotor_fault_start_time = float(payload.get("rotor_fault_start_time", np.nan))
        rotor_fault_state = str(payload.get("rotor_fault_state", "disabled"))
        controller_status_label = str(payload.get("controller_status_label", ""))
        controller_status_value = float(payload.get("controller_status_value", np.nan))
        motor_thrusts = _rotor_vector_or_none(payload.get("motor_thrusts"))
        effectiveness_truth = _rotor_vector_or_none(payload.get("effectiveness_truth"))
        effectiveness_estimate = _rotor_vector_or_none(payload.get("effectiveness_estimate"))
        return frame, sim_time, wind_force, yaw_rate_dps, position_error_norm, position_error_norm_for_max, position_error_max_payload, waiting_for_wind, position_error_waiting_label, controller_mode, rotor_fault_case, rotor_fault_rotor, rotor_fault_start_time, rotor_fault_state, controller_status_label, controller_status_value, motor_thrusts, effectiveness_truth, effectiveness_estimate
    frame = np.asarray(payload)
    return frame, np.nan, np.zeros(3), np.nan, np.nan, np.nan, np.nan, False, "", "n/a", "", 0, np.nan, "disabled", "", np.nan, None, None, None

def _update_telemetry_plots(sim_time, wind_force, yaw_rate_dps, position_error_norm,
                            motor_thrusts, effectiveness_truth, effectiveness_estimate):
    global force_count, time_history, wind_force_history, yaw_rate_history
    if not np.isfinite(sim_time):
        return

    time_history[force_count] = sim_time
    wind_force_history[force_count] = wind_force
    yaw_rate_history[force_count] = yaw_rate_dps
    pos_err_history[force_count] = position_error_norm
    if motor_thrusts is not None:
        motor_thrust_history[force_count] = motor_thrusts
    if effectiveness_estimate is not None:
        eff_est_history[force_count] = effectiveness_estimate
    if effectiveness_truth is not None:
        eff_truth_history[force_count] = effectiveness_truth
    if force_count < np.size(time_history, 0) - 1:
        force_count += 1

    if force_count < 2:
        return

    t = time_history[:force_count]
    t_end = t[-1]
    t_start = max(0.0, t_end - 10.0)
    x_max = max(10.0, t_end)

    err = pos_err_history[:force_count]
    err_line.set_data(t, err)
    ax_err.set_xlim(t_start, x_max)
    finite_err = err[np.isfinite(err)]
    err_lim = 0.05 if finite_err.size == 0 else max(0.05, 1.2 * float(np.max(finite_err)))
    ax_err.set_ylim(0.0, err_lim)

    yaw_rate = yaw_rate_history[:force_count]
    yaw_rate_line.set_data(t, yaw_rate)
    ax_yaw.set_xlim(t_start, x_max)
    finite_yaw = yaw_rate[np.isfinite(yaw_rate)]
    if finite_yaw.size == 0:
        yaw_lim = 10.0
    else:
        yaw_lim = max(10.0, 1.2 * float(np.max(np.abs(finite_yaw))))
    ax_yaw.set_ylim(-yaw_lim, yaw_lim)

    if ax_force is not None:
        f = wind_force_history[:force_count]
        force_line_x.set_data(t, f[:,0])
        force_line_y.set_data(t, f[:,1])
        force_line_z.set_data(t, f[:,2])
        ax_force.set_xlim(t_start, x_max)
        max_abs = np.max(np.abs(f))
        y_lim = max(0.03, 1.2 * max_abs)
        ax_force.set_ylim(-y_lim, y_lim)

    if ax_rotor is not None:
        ax_rotor.set_xlim(t_start, x_max)
        if rotor_plot_kind == "effectiveness":
            est = eff_est_history[:force_count]
            truth = eff_truth_history[:force_count]
            for i in range(4):
                rotor_lines[i].set_data(t, est[:, i])
                rotor_truth_lines[i].set_data(t, truth[:, i])
        else:
            thrusts = motor_thrust_history[:force_count]
            for i in range(4):
                rotor_lines[i].set_data(t, thrusts[:, i])
            finite_thrust = thrusts[np.isfinite(thrusts)]
            thrust_lim = 0.05 if finite_thrust.size == 0 else max(0.05, 1.2 * float(np.max(finite_thrust)))
            ax_rotor.set_ylim(0.0, thrust_lim)

def set_frame(payload):
    global status_text, controller_status_text, position_error_max
    frame, sim_time, wind_force, yaw_rate_dps, position_error_norm, position_error_norm_for_max, position_error_max_payload, waiting_for_wind, position_error_waiting_label, controller_mode, rotor_fault_case, rotor_fault_rotor, rotor_fault_start_time, rotor_fault_state, controller_status_label, controller_status_value, motor_thrusts, effectiveness_truth, effectiveness_estimate = _parse_payload(payload)

    if frame.shape != (3, 6):
        raise ValueError("Frame must be a 3x6 matrix.")

    # convert 3x6 world_frame matrix into three line_data objects which is 3x2 (row:point index, column:x,y,z)
    lines_data = [frame[:,[0,2]], frame[:,[1,3]], frame[:,[4,5]]]
    for line, line_data in zip(body_lines, lines_data):
        x, y, z = line_data
        line.set_data(x, y)
        line.set_3d_properties(z)

    global history, history_time, trajectory_history_window_s
    history.append(frame[:,4].copy())
    history_time.append(sim_time)
    if trajectory_history_window_s is not None and np.isfinite(sim_time):
        while history_time and np.isfinite(history_time[0]) and (sim_time - history_time[0]) > trajectory_history_window_s:
            history.popleft()
            history_time.popleft()
    if history:
        history_arr = np.asarray(history, dtype=float)
        history_line.set_data(history_arr[:,0], history_arr[:,1])
        history_line.set_3d_properties(history_arr[:,2])
    else:
        history_line.set_data([], [])
        history_line.set_3d_properties([])
    # ax.plot3D(xline, yline, zline, 'blue')

    # Current position and attitude display.
    origin = frame[:,4]
    _expand_3d_limits_to_include(origin)
    # m1 front-right, m2 rear-right, m3 rear-left, m4 front-left (+x forward, +y left).
    x_axis = 0.5 * (frame[:,0] + frame[:,3]) - 0.5 * (frame[:,1] + frame[:,2])
    y_axis = 0.5 * (frame[:,2] + frame[:,3]) - 0.5 * (frame[:,0] + frame[:,1])
    x_norm = np.linalg.norm(x_axis)
    y_norm = np.linalg.norm(y_axis)
    if x_norm > 1e-9 and y_norm > 1e-9:
        x_hat = x_axis / x_norm
        y_hat = y_axis / y_norm
        z_hat = np.cross(x_hat, y_hat)
        z_norm = np.linalg.norm(z_hat)
        if z_norm > 1e-9:
            z_hat = z_hat / z_norm
            # RotToRPY inverts RPYToRot (world->body): body axes are its rows.
            rot = np.vstack((x_hat, y_hat, z_hat))
            roll, pitch, yaw = RotToRPY(rot)
            roll_deg, pitch_deg, yaw_deg = np.degrees([roll, pitch, yaw])
        else:
            roll_deg, pitch_deg, yaw_deg = 0.0, 0.0, 0.0
    else:
        roll_deg, pitch_deg, yaw_deg = 0.0, 0.0, 0.0

    _update_telemetry_plots(
        sim_time,
        wind_force,
        yaw_rate_dps,
        position_error_norm,
        motor_thrusts,
        effectiveness_truth,
        effectiveness_estimate,
    )

    time_str = "n/a" if not np.isfinite(sim_time) else "{:.2f}".format(sim_time)
    yaw_rate_text = "n/a" if not np.isfinite(yaw_rate_dps) else "{:.1f}".format(yaw_rate_dps)
    pos_error_text = ""
    if np.isfinite(position_error_max_payload):
        position_error_max = position_error_max_payload
    elif np.isfinite(position_error_norm_for_max):
        if not np.isfinite(position_error_max):
            position_error_max = position_error_norm_for_max
        else:
            position_error_max = max(position_error_max, position_error_norm_for_max)
    if np.isfinite(position_error_norm):
        if waiting_for_wind:
            wait_label = position_error_waiting_label or "event"
            pos_error_text = "\n|position error|: {:.3f} m    max: waiting for {}".format(
                position_error_norm, wait_label
            )
        elif np.isfinite(position_error_max):
            pos_error_text = "\n|position error|: {:.3f} m    max: {:.3f} m".format(
                position_error_norm, position_error_max
            )
        else:
            pos_error_text = "\n|position error|: {:.3f} m    max: n/a".format(
                position_error_norm
            )
    if controller_status_text is not None:
        status_suffix = ""
        if controller_status_label:
            status_suffix = "  ·  " + controller_status_label.replace("_", " ")
            if not controller_status_label.isupper() and np.isfinite(controller_status_value):
                status_suffix += ": {:.3g}".format(controller_status_value)
        controller_status_text.set_text("Controller: {}{}".format(controller_mode, status_suffix))

    if status_text is not None:
        wind_text = ""
        if show_wind_panels:
            wind_units = "m/s" if wind_as_velocity else "N"
            wind_norm = np.linalg.norm(wind_force)
            wind_text = "\nWind [{}]: Fx={:.3f}, Fy={:.3f}, Fz={:.3f}, |F|={:.3f}".format(
                wind_units, wind_force[0], wind_force[1], wind_force[2], wind_norm
            )
        rotor_fault_text = ""
        if rotor_fault_case:
            fault_start_text = "n/a"
            if np.isfinite(rotor_fault_start_time):
                fault_start_text = "{:.2f}".format(rotor_fault_start_time)
            rotor_fault_text = (
                "\nRotor fault: {}  rotor={}  start={} s  state={}".format(
                    rotor_fault_case,
                    rotor_fault_rotor,
                    fault_start_text,
                    rotor_fault_state,
                )
            )
        status_text.set_text(
            "t: {} s\nPos: x={:.2f}, y={:.2f}, z={:.2f}\nRoll: {:.1f} deg  Pitch: {:.1f} deg  Yaw: {:.1f} deg  Yaw rate: {} deg/s{}{}{}".format(
                time_str, origin[0], origin[1], origin[2], roll_deg, pitch_deg, yaw_deg,
                yaw_rate_text, wind_text, pos_error_text, rotor_fault_text
            )
        )
