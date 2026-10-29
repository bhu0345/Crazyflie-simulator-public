from collections import deque
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

from crazyflie_sim.utils.utils import RotToRPY


# Dark "ground control" palette shared by the live view and the launcher.
THEME = {
    "bg": "#0b1018",
    "scene": "#0d1420",
    "surface": "#121a26",
    "surface_alt": "#172131",
    "hover": "#1d2a3d",
    "border": "#1f2a3a",
    "border_strong": "#2b3a50",
    "text": "#e6edf3",
    "muted": "#8b98ac",
    "faint": "#8b98ac",
    "accent": "#5aa2ff",
    "accent_strong": "#2f6feb",
    "success": "#3dd68c",
    "warn": "#f5a524",
    "danger": "#f2555a",
    "mono": "'Menlo', 'Consolas', 'DejaVu Sans Mono', monospace",
}
RUN_STATE_COLORS = {
    "READY": THEME["muted"],
    "RUNNING": THEME["success"],
    "PAUSED": THEME["warn"],
    "DONE": THEME["accent"],
    "ERROR": THEME["danger"],
}


def _rgb255(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _gl_color(color, alpha=1.0):
    return tuple(c / 255.0 for c in _rgb255(color)) + (alpha,)


def _gl_mix(color, background, amount):
    """Opaque GL colour: ``color`` laid over ``background`` at ``amount``.

    Opaque lines keep the framebuffer alpha at 1, so the view composites the
    same on every platform."""
    fg, bg = _rgb255(color), _rgb255(background)
    return tuple((amount * f + (1.0 - amount) * b) / 255.0 for f, b in zip(fg, bg)) + (1.0,)


def _make_world_indicator(view, QtCore, QtWidgets):
    # Keep Qt optional for headless callers importing this module's helpers.
    from pyqtgraph.Qt import QtGui

    class WorldIndicator(QtWidgets.QWidget):
        def __init__(self):
            super().__init__(view)
            self.axes = np.eye(3)
            self.setFixedSize(96, 104)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.setAccessibleName("World orientation")
            self.setAccessibleDescription("World axes viewed from the current camera; independent of vehicle attitude")

        def set_axes(self, axes):
            if not np.allclose(self.axes, axes):
                self.axes = np.asarray(axes)
                self.update()

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtGui.QColor(THEME["border"]))
            painter.setBrush(QtGui.QColor(11, 16, 24, 210))
            painter.drawRoundedRect(QtCore.QRectF(0.5, 0.5, 95, 103), 8, 8)
            font = painter.font()
            font.setPixelSize(12)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(THEME["muted"]))
            painter.drawText(QtCore.QRectF(0, 7, 96, 18), QtCore.Qt.AlignmentFlag.AlignCenter, "World")
            center = QtCore.QPointF(48, 62)
            colors = ("#f2555a", "#3dd68c", "#5aa2ff")
            # Draw the nearer axes last; only camera rotation enters here.
            for i in np.argsort(self.axes[:, 2]):
                x, y, _z = self.axes[i]
                direction = QtCore.QPointF(float(x), float(-y))
                tip = center + direction * 25
                color = QtGui.QColor(colors[i])
                painter.setPen(QtGui.QPen(color, 1))
                painter.drawLine(center, tip)
                painter.setBrush(color)
                painter.drawEllipse(tip, 2, 2)
                label = center + direction * 35
                painter.drawText(QtCore.QRectF(label.x() - 7, label.y() - 8, 14, 16),
                                 QtCore.Qt.AlignmentFlag.AlignCenter, "XYZ"[i])
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(THEME["muted"]))
            painter.drawEllipse(center, 2, 2)

    indicator = WorldIndicator()
    indicator.show()
    return indicator


def _pill_qss(color):
    """A tinted status pill: text in ``color`` on a faint fill of it."""
    r, g, b = _rgb255(color)
    return (
        "QLabel {{ color: {c}; background-color: rgba({r}, {g}, {b}, 38);"
        # Qt drops the rounding when the radius exceeds half the height.
        " border: 1px solid rgba({r}, {g}, {b}, 96); border-radius: 9px;"
        " padding: 3px 10px; font-size: 12px; font-weight: 700;"
        " letter-spacing: 0.6px; }}".format(c=color, r=r, g=g, b=b)
    )


def scroll_qss():
    """Thin dark scrollbars and tooltips, shared by the panel and launcher."""
    return """
        QToolTip {{
            color: {text}; background-color: {surface_alt};
            border: 1px solid {border_strong}; border-radius: 6px; padding: 6px 8px;
        }}
        QScrollArea {{ background: transparent; border: none; }}
        QScrollArea > QWidget > QWidget {{ background: transparent; }}
        QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
        QScrollBar::handle:vertical {{
            background: {border_strong}; border-radius: 3px; min-height: 36px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {faint}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
    """.format(**THEME)


def _configure_qt_environment():
    """Keep PyQtGraph on PyQt6 and its bundled plugins.

    A Conda installation can leave a Qt 5 ``qt.conf`` next to the Python
    executable.  Qt 6 then reads that file and searches Conda's Qt 5 plugin
    directory instead of the plugins bundled with PyQt6.  On macOS this
    presents as a missing ``cocoa`` platform plugin even though the correct
    plugin is installed inside the PyQt6 package.
    """
    pyqt6_spec = importlib.util.find_spec("PyQt6")
    if pyqt6_spec is None:
        return

    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

    if sys.platform != "darwin" or not pyqt6_spec.submodule_search_locations:
        return

    pyqt6_dir = Path(next(iter(pyqt6_spec.submodule_search_locations)))
    plugin_dir = pyqt6_dir / "Qt6" / "plugins"
    cocoa_plugin = plugin_dir / "platforms" / "libqcocoa.dylib"
    if not cocoa_plugin.is_file():
        return

    plugin_dir_text = str(plugin_dir)
    configured_paths = [
        path
        for path in os.environ.get("QT_PLUGIN_PATH", "").split(os.pathsep)
        if path
    ]
    if plugin_dir_text not in configured_paths:
        os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(
            [plugin_dir_text, *configured_paths]
        )


def _load_qt_modules():
    _configure_qt_environment()
    try:
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtCore, QtWidgets
    except ImportError as exc:
        raise ImportError(
            "PyQtGraph/OpenGL visualization requires pyqtgraph, PyOpenGL, "
            "and a Qt binding such as PyQt6. Install them with:\n"
            "    python -m pip install pyqtgraph PyOpenGL PyQt6"
        ) from exc
    return pg, gl, QtCore, QtWidgets


def _qt_exec(app):
    if hasattr(app, "exec"):
        return app.exec()
    return app.exec_()


def _as_waypoint_positions(waypoints):
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    if waypoints.shape[1] > 3:
        waypoints = waypoints[:, :3]
    return waypoints.reshape((-1, 3))


def _as_rotor_vector(value):
    """(4,) float array or None when the payload does not carry the field."""
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(4,)


def _parse_payload(payload):
    if isinstance(payload, dict):
        frame = np.asarray(payload.get("frame"), dtype=float)
        sim_time = float(payload.get("time", np.nan))
        wind_force = np.asarray(
            payload.get("wind_force_world", np.zeros(3)), dtype=float
        ).reshape(3,)
        yaw_rate_dps = float(payload.get("yaw_rate_dps", np.nan))
        position_error_norm = float(payload.get("position_error_norm", np.nan))
        position_error_norm_for_max = float(
            payload.get("position_error_norm_for_max", position_error_norm)
        )
        position_error_max_payload = float(payload.get("position_error_max", np.nan))
        waiting_for_wind = bool(payload.get("waiting_for_wind", False))
        position_error_waiting_label = str(
            payload.get("position_error_waiting_label", "")
        )
        controller_mode = str(payload.get("controller_mode", "n/a"))
        controller_status_label = str(payload.get("controller_status_label", ""))
        controller_status_value = float(
            payload.get("controller_status_value", np.nan)
        )
        rotor_fault_case = str(payload.get("rotor_fault_case", ""))
        rotor_fault_rotor = int(payload.get("rotor_fault_rotor", 0))
        rotor_fault_start_time = float(payload.get("rotor_fault_start_time", np.nan))
        rotor_fault_state = str(payload.get("rotor_fault_state", "disabled"))
        attitude_deg = np.asarray(
            payload.get("attitude_deg", (np.nan, np.nan, np.nan)), dtype=float
        ).reshape(3,)
        motor_thrusts = _as_rotor_vector(payload.get("motor_thrusts"))
        effectiveness_truth = _as_rotor_vector(payload.get("effectiveness_truth"))
        effectiveness_estimate = _as_rotor_vector(
            payload.get("effectiveness_estimate")
        )
        return (
            frame,
            sim_time,
            wind_force,
            yaw_rate_dps,
            position_error_norm,
            position_error_norm_for_max,
            position_error_max_payload,
            waiting_for_wind,
            position_error_waiting_label,
            controller_mode,
            rotor_fault_case,
            rotor_fault_rotor,
            rotor_fault_start_time,
            rotor_fault_state,
            attitude_deg,
            controller_status_label,
            controller_status_value,
            motor_thrusts,
            effectiveness_truth,
            effectiveness_estimate,
        )

    frame = np.asarray(payload, dtype=float)
    return (
        frame,
        np.nan,
        np.zeros(3, dtype=float),
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        False,
        "",
        "n/a",
        "",
        0,
        np.nan,
        "disabled",
        np.full(3, np.nan),
        "",
        np.nan,
        None,
        None,
        None,
    )


def _fmt(value, spec, fallback="—"):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(number):
        return fallback
    return format(number, spec)


def _pretty_controller(name):
    from crazyflie_sim.controllers import registry

    try:
        return registry.get(name).label
    except ValueError:
        return str(name or "Not available").replace("_", " ")


def _format_status(label, value):
    """Controller status readout: enum-style labels stand alone, scalar
    diagnostics show their value."""
    text = str(label or "").strip()
    if not text:
        return "—"
    pretty = text.replace("_", " ")
    if text.isupper():
        return pretty
    try:
        number = float(value)
    except (TypeError, ValueError):
        return pretty
    if not np.isfinite(number):
        return pretty
    return "{}: {:.3g}".format(pretty, number)


def _frame_attitude_degrees(frame):
    # Same motor-pair convention as _frame_rotation_and_origin: m1 front-right,
    # m2 rear-right, m3 rear-left, m4 front-left (+x forward, +y left).
    x_axis = 0.5 * (frame[:, 0] + frame[:, 3]) - 0.5 * (frame[:, 1] + frame[:, 2])
    y_axis = 0.5 * (frame[:, 2] + frame[:, 3]) - 0.5 * (frame[:, 0] + frame[:, 1])
    x_norm = np.linalg.norm(x_axis)
    y_norm = np.linalg.norm(y_axis)
    if x_norm <= 1e-9 or y_norm <= 1e-9:
        return 0.0, 0.0, 0.0

    x_hat = x_axis / x_norm
    y_hat = y_axis / y_norm
    z_hat = np.cross(x_hat, y_hat)
    z_norm = np.linalg.norm(z_hat)
    if z_norm <= 1e-9:
        return 0.0, 0.0, 0.0

    # RotToRPY inverts RPYToRot, which maps world->body: its ROWS are the
    # body axes expressed in world coordinates, so stack them as rows.
    rot = np.row_stack((x_hat, y_hat, z_hat / z_norm))
    return tuple(np.degrees(RotToRPY(rot)))


def _frame_rotation_and_origin(frame):
    """Orthonormal body->world rotation and origin from a 3x6 world frame.

    Motor layout (model/params.py body_frame): m1 front-right, m2 rear-right,
    m3 rear-left, m4 front-left (+x forward, +y left). So the front pair is
    (m1, m4), the rear pair (m2, m3), the left pair (m3, m4).
    """
    x_axis = 0.5 * (frame[:, 0] + frame[:, 3]) - 0.5 * (frame[:, 1] + frame[:, 2])
    y_axis = 0.5 * (frame[:, 2] + frame[:, 3]) - 0.5 * (frame[:, 0] + frame[:, 1])
    x_norm = np.linalg.norm(x_axis)
    y_norm = np.linalg.norm(y_axis)
    if x_norm <= 1e-9 or y_norm <= 1e-9:
        return np.eye(3), frame[:, 4].copy()
    x_hat = x_axis / x_norm
    y_hat = y_axis / y_norm
    z_hat = np.cross(x_hat, y_hat)
    z_norm = np.linalg.norm(z_hat)
    if z_norm <= 1e-9:
        return np.eye(3), frame[:, 4].copy()
    return np.column_stack((x_hat, y_hat, z_hat / z_norm)), frame[:, 4].copy()


# ---------------------------------------------------------------------------
# Body mesh builders (body frame, z up, +x forward; meters)
# ---------------------------------------------------------------------------


def _box_mesh(sx, sy, sz, center=(0.0, 0.0, 0.0), rot_z_deg=0.0):
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    verts = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # -y
            [1, 5, 6], [1, 6, 2],  # +x
            [2, 6, 7], [2, 7, 3],  # +y
            [3, 7, 4], [3, 4, 0],  # -x
        ],
        dtype=np.int32,
    )
    if rot_z_deg:
        a = np.radians(rot_z_deg)
        c, s = np.cos(a), np.sin(a)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        verts = verts @ rot.T
    return verts + np.asarray(center, dtype=float), faces


def _cylinder_mesh(radius, height, center=(0.0, 0.0, 0.0), sides=18):
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    bottom = np.column_stack((ring, np.full(sides, -height / 2.0)))
    top = np.column_stack((ring, np.full(sides, height / 2.0)))
    verts = np.vstack((bottom, top))
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, sides + j])
        faces.append([i, sides + j, sides + i])
        faces.append([0, j, i])  # bottom cap fan (rough but shaded flat)
        faces.append([sides, sides + i, sides + j])  # top cap fan
    return verts + np.asarray(center, dtype=float), np.asarray(faces, dtype=np.int32)


def _disc_mesh(radius, center=(0.0, 0.0, 0.0), sides=36):
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.zeros(sides))
    )
    verts = np.vstack(([[0.0, 0.0, 0.0]], ring))
    faces = np.array(
        [[0, i + 1, (i + 1) % sides + 1] for i in range(sides)], dtype=np.int32
    )
    return verts + np.asarray(center, dtype=float), faces


def _spin_verts(verts, center, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (verts - center) @ rot.T + center


def _sphere_mesh(radius, center=(0.0, 0.0, 0.0), segments=16, rings=10):
    """UV sphere; renders through the mesh shader path (reliable colors)."""
    verts = []
    for i in range(rings + 1):
        phi = np.pi * i / rings
        for j in range(segments):
            theta = 2.0 * np.pi * j / segments
            verts.append(
                [
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi),
                ]
            )
    faces = []
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + (j + 1) % segments
            d = (i + 1) * segments + j
            faces.append([a, c, d])  # outward winding (correct lighting)
            faces.append([a, b, c])
    return (
        np.asarray(verts, dtype=float) + np.asarray(center, dtype=float),
        np.asarray(faces, dtype=np.int32),
    )


def _grid_lines(half_size, spacing):
    """Grid line segments for GLLinePlotItem(mode='lines').

    GLGridItem uses immediate-mode glBegin/glEnd, which renders nothing in
    this Qt6/OpenGL context -- build the same grid from line segments.
    """
    vals = np.arange(-half_size, half_size + spacing * 0.001, spacing)
    segs = []
    for v in vals:
        segs.append([[v, -half_size, 0.0], [v, half_size, 0.0]])
        segs.append([[-half_size, v, 0.0], [half_size, v, 0.0]])
    return np.asarray(segs, dtype=float).reshape(-1, 3)


def _dashify(points, dash_len, gap_len):
    """Split a polyline into dash segments (Nx3 pairs for mode='lines').

    GL lines have no native dash style; cut the path into dash_len pieces
    separated by gap_len gaps instead.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.empty((0, 3), dtype=float)
    out = []
    on = True
    left = dash_len  # length left in the current dash/gap piece
    piece_start = pts[0]
    cur = pts[0]
    for nxt in pts[1:]:
        vec = nxt - cur
        seg_len = float(np.linalg.norm(vec))
        if seg_len <= 1e-12:
            continue
        pos = 0.0  # distance consumed along this segment
        while seg_len - pos >= left:
            cut = cur + vec * ((pos + left) / seg_len)
            if on:
                out.append((piece_start, cut))
            piece_start = cut
            pos += left
            on = not on
            left = dash_len if on else gap_len
        left -= seg_len - pos
        cur = nxt
    if on:
        out.append((piece_start, pts[-1]))
    return np.asarray(out, dtype=float).reshape(-1, 3)


class _PyQtGraphQuadWindow:
    COLOR_BG = THEME["scene"]
    # Opaque scene lines (see _gl_mix): faint minor grid, brighter 1 m grid.
    COLOR_GRID_MINOR = _gl_mix("#5b6b82", THEME["scene"], 0.22)
    COLOR_GRID_MAJOR = _gl_mix("#8b98ac", THEME["scene"], 0.32)
    COLOR_TRAIL = _gl_color("#5cc8ff")
    COLOR_WIND_ARROW = "#e879f9"
    _PANEL_QSS = """
        QWidget#simRoot {{ background-color: {bg}; }}
        QFrame#sidePanel {{ background-color: {bg}; border: none; }}
        QFrame#telemetryCard, QFrame#plotCard {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 12px;
        }}
        QFrame#faultCard {{
            background-color: rgba(245, 165, 36, 26);
            border: 1px solid rgba(245, 165, 36, 110);
            border-radius: 12px;
        }}
        QLabel {{ color: {text}; }}
        QLabel#brandTitle {{ color: {text}; font-size: 15px; font-weight: 700; }}
        QLabel#heroValue {{
            color: {text};
            font-size: 24px;
            font-weight: 600;
            font-family: {mono};
        }}
        QLabel#heroCaption {{
            color: {faint};
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.8px;
        }}
        QLabel#cardTitle {{
            color: {muted};
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.8px;
        }}
        QLabel#kvKey {{ color: {muted}; font-size: 12px; }}
        QLabel#kvVal {{
            color: {text};
            font-size: 12.5px;
            font-weight: 500;
            font-family: {mono};
        }}
        QLabel#axisHeader {{ color: {faint}; font-size: 12px; font-weight: 700; }}
        QLabel#rowKey {{ color: {muted}; font-size: 12px; font-weight: 600; }}
        QLabel#faultTitle {{
            color: {warn}; font-size: 12px; font-weight: 800; letter-spacing: 0.8px;
        }}
        QLabel#faultDetail {{ color: #f8d58a; font-size: 12px; font-weight: 600; }}
    """.format(**THEME) + scroll_qss()
    _PANEL_QSS += """
        QWidget { color: %(text)s; }
        QPushButton, QToolButton, QComboBox, QDoubleSpinBox {
            color: %(text)s; background: %(surface_alt)s;
            border: 1px solid %(border_strong)s; border-radius: 6px;
            padding: 5px; font-size: 12px;
        }
        QPushButton:hover, QPushButton:checked, QToolButton:hover { background: %(hover)s; }
        QToolButton { padding-right: 20px; }
        QToolButton::menu-indicator {
            subcontrol-origin: padding; subcontrol-position: center right;
            width: 10px; height: 6px; right: 5px;
        }
        QPushButton:focus, QToolButton:focus, QComboBox:focus, QDoubleSpinBox:focus { border: 1px solid %(accent)s; }
        QMenu { color: %(text)s; background: %(surface_alt)s; border: 1px solid %(border_strong)s; padding: 4px; }
        QMenu::item { padding: 6px 18px; }
        QMenu::item:selected { background: %(hover)s; }
        QComboBox QAbstractItemView {
            color: %(text)s; background: %(surface_alt)s;
            selection-background-color: %(accent_strong)s;
        }
        QCheckBox { color: %(text)s; font-size: 12px; }
        QProgressBar { background: %(border)s; border: none; border-radius: 3px; max-height: 6px; }
        QProgressBar::chunk { background: %(accent)s; border-radius: 3px; }
        QSplitter::handle { background: %(bg)s; }
        QSplitter::handle:hover { background: %(border_strong)s; }
        QDialog { background: %(surface)s; }
    """ % THEME
    TRAIL_DASH_M = 0.010  # dash length along the trail
    TRAIL_GAP_M = 0.012  # gap length along the trail

    # Per-rotor trace colors (m1..m4) shared by the rotor panels.
    ROTOR_COLORS = ("#ff8a3d", "#5aa2ff", "#3dd68c", "#b18cff")

    # Visual scale of the quad body meshes.  1.0 = true scale (0.10 m
    # motor-to-motor Crazyflie frame), which is a few pixels in a metre-scale
    # scene; the body is drawn larger, positions and the trail are not.
    BODY_SCALE = 2.5

    # Crazyflie-scale body geometry (meters, body frame: +x forward, z up).
    _XY = 0.05 / np.sqrt(2.0)  # motor x/y offset (arm_length / sqrt(2))
    _ARM_LEN = 0.10  # motor-to-motor diagonal
    _MOTOR_R = 0.0042
    _MOTOR_H = 0.007
    _PROP_R = 0.023
    _PROP_Z = 0.0085
    _PROP_DIRS = (1.0, -1.0, 1.0, -1.0)  # m1/m3 CCW, m2/m4 CW (viewed from above)

    def __init__(
        self,
        waypoints,
        get_world_frame,
        frames,
        history_window_s,
        fps,
        pg,
        gl,
        QtCore,
        QtWidgets,
        view_config=None,
    ):
        self.pg = pg
        self.gl = gl
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets

        self.waypoints = _as_waypoint_positions(waypoints)
        self.get_world_frame = get_world_frame
        self.frames = int(frames)
        self.history_window_s = history_window_s
        self.fps = max(1.0, float(fps))
        self.frame_index = 0

        # Static layout switches.  No view_config (legacy callers) keeps the
        # historical layout with every panel visible.
        self.view_config = dict(view_config or {})
        self.sim_mode = str(self.view_config.get("sim_mode", "") or "")
        self.show_wind = self.sim_mode in ("", "hover_wind")
        self.show_rotor = self.sim_mode == "hover_rotor_fault"
        self.wind_as_velocity = bool(self.view_config.get("wind_as_velocity", False))
        self.wind_formula = str(self.view_config.get("wind_formula", "") or "")
        self.rotor_plot_kind = (
            str(self.view_config.get("rotor_plot", "thrust"))
            if self.show_rotor
            else ""
        )
        event_time = self.view_config.get("event_time_s")
        self.event_time_s = None if event_time is None else float(event_time)
        self.event_label = str(self.view_config.get("event_label", "") or "")

        self.duration_s = float(self.view_config.get("duration_s", self.frames / self.fps))
        self.event_description = self.view_config.get("event_description", self.event_label)
        self._plots = []
        self._expanded = []
        self._setting_ranges = False
        self._cursor_follow = True
        self._last_payload = None
        self.position_error_max = np.nan
        self.history = deque()
        self.history_time = deque()
        self.time_history = []
        self.wind_force_history = []
        self.yaw_rate_history = []
        self.pos_err_history = []
        self.motor_thrust_history = []
        self.eff_est_history = []
        self.eff_truth_history = []
        self.prop_angle = 0.0
        self._ui_phase = 0

        self.window = QtWidgets.QWidget()
        self.window.setObjectName("simRoot")
        self.window.setWindowTitle("Crazyflie Simulator")
        self.window.setStyleSheet(self._PANEL_QSS)
        self._run_complete = False
        self._build_ui()
        self._init_scene()

        self.timer = QtCore.QTimer(self.window)
        self.timer.timeout.connect(self._on_timer)
        self.timer.setInterval(max(1, int(round(1000.0 / self.fps))))
        self.playback_speed.currentIndexChanged.connect(self._set_playback_speed)

    def _build_ui(self):
        W, C = self.QtWidgets, self.QtCore
        self.pg.setConfigOptions(antialias=True, background=THEME["surface"], foreground=THEME["muted"])
        layout = W.QHBoxLayout(self.window)
        layout.setContentsMargins(16, 16, 16, 16)
        self.splitter = W.QSplitter(C.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        self.splitter.setAccessibleName("Scene and telemetry divider")
        layout.addWidget(self.splitter)
        self.view = self.gl.GLViewWidget()
        self.view.setMinimumWidth(320)
        self.view.setBackgroundColor(self.COLOR_BG)
        self.splitter.addWidget(self.view)
        panel = W.QFrame()
        panel.setObjectName("sidePanel")
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(W.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(C.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(340)
        scroll.setWidget(panel)
        self.splitter.addWidget(scroll)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([980, 420])
        side = W.QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 6, 0)
        side.setSpacing(10)
        self.brand = self._label("Crazyflie Simulator", "brandTitle")
        side.addWidget(self.brand)

        hero, content = self._card("telemetryCard")
        heading = W.QHBoxLayout()
        heading.addWidget(self._label("Position error", "heroCaption"))
        heading.addStretch()
        self.run_state = W.QLabel()
        heading.addWidget(self.run_state)
        content.addLayout(heading)
        self.error_value = self._label("Not available", "heroValue")
        content.addWidget(self.error_value)
        self.time_value = self._label("0.00 / {:.2f} s".format(self.duration_s), "kvKey")
        time_row = W.QHBoxLayout()
        time_row.addWidget(self.time_value, 1)
        self.playback_speed = W.QComboBox()
        for label, speed in (("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0)):
            self.playback_speed.addItem(label, speed)
        self.playback_speed.setCurrentIndex(2)
        self.playback_speed.setAccessibleName("Playback speed")
        self.playback_speed.setToolTip("Playback speed: slows the display clock without changing the simulation timestep or results")
        time_row.addWidget(self.playback_speed)
        content.addLayout(time_row)
        self.progress = W.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("Simulation progress")
        content.addWidget(self.progress)
        self.fault_card, event_layout = self._card("faultCard")
        self.fault_title = self._label("Event", "faultTitle")
        self.fault_detail = self._wrapping_label("")
        self.fault_detail.setObjectName("faultDetail")
        event_layout.addWidget(self.fault_title)
        event_layout.addWidget(self.fault_detail)
        content.addWidget(self.fault_card)
        side.addWidget(hero)
        self.set_run_state("Running")
        self._refresh_event(0.0)

        state_card, state_layout = self._card("telemetryCard")
        self._xyz_values = {}
        rows = [("pos", "Position (m)", ("X", "Y", "Z")),
                ("att", "Attitude (deg)", ("Roll", "Pitch", "Yaw"))]
        if self.show_wind:
            rows.append(("wind", "Wind velocity (m/s)" if self.wind_as_velocity else "Wind force (N)", ("X", "Y", "Z")))
        for key, title, headers in rows:
            state_layout.addWidget(self._label(title, "rowKey"))
            grid = W.QGridLayout()
            self._xyz_values[key] = []
            for col, name in enumerate(headers):
                grid.addWidget(self._label(name, "axisHeader"), 0, col)
                value = self._label("Not available", "kvVal")
                grid.addWidget(value, 1, col)
                grid.setColumnStretch(col, 1)
                self._xyz_values[key].append(value)
            state_layout.addLayout(grid)
        stats = W.QGridLayout()
        self._stat_values = {"pos_err": self.error_value}
        items = [("yaw_rate", "Angular velocity (world Z)"),
                 ("pos_err_max", "Peak error after event" if self.event_time_s is not None else "Peak error")]
        if self.show_wind:
            items.append(("wind_norm", "Wind magnitude"))
        for row, (key, caption) in enumerate(items):
            stats.addWidget(self._label(caption, "kvKey"), row, 0)
            value = self._label("Not available", "kvVal")
            value.setAlignment(C.Qt.AlignmentFlag.AlignRight)
            stats.addWidget(value, row, 1)
            self._stat_values[key] = value
        state_layout.addLayout(stats)
        side.addWidget(state_card)

        self.result_card, result_layout = self._card("telemetryCard")
        result_layout.addWidget(self._label("Recorded result", "cardTitle"))
        self.result_text = self._wrapping_label("")
        result_layout.addWidget(self.result_text)
        self.export_result = W.QPushButton("Export results…")
        result_layout.addWidget(self.export_result)
        self.result_card.hide()
        side.addWidget(self.result_card)

        card, self.err_plot, curves = self._plot_card("Position error", "m", (("error", THEME["accent"]),), "error")
        self.err_curve = curves[0]
        side.addWidget(card)
        self.force_plot = None
        self.force_x = self.force_y = self.force_z = None
        if self.show_wind:
            card, self.force_plot, curves = self._plot_card(
                "Wind velocity" if self.wind_as_velocity else "Wind force",
                "m/s" if self.wind_as_velocity else "N",
                (("X", "#ff8a3d"), ("Y", "#3dd68c"), ("Z", "#b18cff")), "wind")
            card.setToolTip(self.wind_formula)
            self.force_x, self.force_y, self.force_z = curves
            side.addWidget(card)
        self.rotor_plot = None
        if self.rotor_plot_kind:
            series = tuple((f"m{i + 1}", color) for i, color in enumerate(self.ROTOR_COLORS))
            caption = None
            if self.rotor_plot_kind == "effectiveness":
                series += tuple((f"m{i + 1} true", color, True) for i, color in enumerate(self.ROTOR_COLORS))
                caption = "Solid: estimate. Dashed: true effectiveness."
            card, self.rotor_plot, curves = self._plot_card(
                "Rotor effectiveness" if caption else "Rotor thrust", "" if caption else "N",
                series, "rotor", caption=caption)
            if caption:
                self.rotor_est_curves, self.rotor_truth_curves = curves[:4], curves[4:]
            else:
                self.rotor_thrust_curves = curves
            side.addWidget(card)
        card, self.yaw_plot, curves = self._plot_card(
            "Angular velocity", "deg/s", (("world Z", "#ff6b81"),), "spin",
            caption="World Z component; not the yaw-angle derivative.")
        rate_help = ("World-vertical component of the airframe's angular velocity. "
                     "Positive and negative values indicate opposite directions; "
                     "oscillation around zero does not imply sustained spinning. "
                     "With tilt, this differs from the rate of change of the Yaw angle above.")
        card.setToolTip(rate_help)
        self._stat_values["yaw_rate"].setToolTip(rate_help)
        self.yaw_curve = curves[0]
        side.addWidget(card)

        # Range and cursor controls sit under the charts, which stay above the fold.
        controls, controls_layout = self._card("telemetryCard")
        row = W.QHBoxLayout()
        self.plot_mode = W.QComboBox()
        self.plot_mode.addItems(["Live 10 s", "Full run", "Inspect"])
        self.plot_mode.setAccessibleName("Chart time range mode")
        self.auto_y = W.QCheckBox("Auto Y")
        self.auto_y.setChecked(True)
        self.fit_button = W.QPushButton("Fit")
        self.fit_button.setToolTip("Fit vertical ranges to the visible time interval")
        row.addWidget(self.plot_mode, 1)
        row.addWidget(self.auto_y)
        row.addWidget(self.fit_button)
        controls_layout.addLayout(row)
        cursor_row = W.QHBoxLayout()
        cursor_row.addWidget(self._label("Cursor (s)", "kvKey"))
        self.cursor_time = W.QDoubleSpinBox()
        self.cursor_time.setDecimals(3)
        self.cursor_time.setButtonSymbols(W.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.cursor_time.setKeyboardTracking(False)
        self.cursor_time.setRange(0, 0)
        self.cursor_time.setSingleStep(1.0 / self.fps)
        self.cursor_time.setAccessibleName("Chart cursor time in seconds")
        cursor_row.addWidget(self.cursor_time, 1)
        self.cursor_previous = W.QPushButton("Previous")
        self.cursor_next = W.QPushButton("Next")
        cursor_row.addWidget(self.cursor_previous)
        cursor_row.addWidget(self.cursor_next)
        controls_layout.addLayout(cursor_row)
        side.addWidget(controls)

        diagnostics, details_layout = self._card("telemetryCard")
        self.details_button = W.QPushButton("Diagnostics and setup")
        self.details_button.setCheckable(True)
        details_layout.addWidget(self.details_button)
        self.details_body = W.QWidget()
        details = W.QVBoxLayout(self.details_body)
        details.setContentsMargins(0, 0, 0, 0)
        self.controller_status = self._wrapping_label("Not available")
        self.configuration_details = self._wrapping_label("")
        details.addWidget(self.controller_status)
        details.addWidget(self.configuration_details)
        self.details_body.hide()
        self.details_button.toggled.connect(self.details_body.setVisible)
        details_layout.addWidget(self.details_body)
        side.addWidget(diagnostics)
        side.addStretch()
        self.plot_mode.currentTextChanged.connect(self._plot_mode_changed)
        self.auto_y.toggled.connect(lambda: self._render_plots())
        self.fit_button.clicked.connect(lambda: self._render_plots(fit=True))
        self.cursor_time.valueChanged.connect(self._cursor_changed)
        self.cursor_previous.clicked.connect(lambda: self._move_cursor(-1))
        self.cursor_next.clicked.connect(lambda: self._move_cursor(1))
        self.window.resize(1440, 900)

    def _wrapping_label(self, text):
        label = self._label(text, "kvKey")
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(self.QtWidgets.QSizePolicy.Policy.Ignored, self.QtWidgets.QSizePolicy.Policy.Preferred)
        label.setTextInteractionFlags(self.QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _set_playback_speed(self):
        speed = self.playback_speed.currentData()
        self.timer.setInterval(max(1, int(round(1000.0 / (self.fps * speed)))))

    def _refresh_event(self, sim_time):
        self.fault_card.setVisible(self.event_time_s is not None)
        if self.event_time_s is None:
            return
        remaining = self.event_time_s - sim_time
        ready = self.run_state.text() == "READY"
        active = remaining <= 0 and not ready
        if getattr(self, "_event_active", None) != active:
            self._event_active = active
            self.fault_card.setStyleSheet("" if active else
                "QFrame#faultCard { background: %s; border: 1px solid %s; border-radius: 8px; }"
                % (THEME["surface_alt"], THEME["border_strong"]))
            self.fault_title.setStyleSheet("color: %s; font-size: 12px; font-weight: 700;"
                                          % (THEME["warn"] if active else THEME["accent"]))
            self.fault_detail.setStyleSheet("color: %s; font-size: 12px;"
                                           % ("#f8d58a" if active else THEME["muted"]))
        self.fault_title.setText("Event active" if active else "Planned event")
        state = "At start" if ready and remaining <= 0 else "Occurred" if active else f"In {remaining:.3f} s"
        self.fault_detail.setText(f"{self.event_description} · {state} · at {self.event_time_s:.3f} s")

    def set_run_state(self, state):
        """Show a run state (Ready, Running, Paused, Done, Error) as a pill."""
        text = str(state).upper()
        if text == self.run_state.text():
            return  # called every frame; restyling is not free
        self.run_state.setText(text)
        self.run_state.setStyleSheet(_pill_qss(RUN_STATE_COLORS.get(text, THEME["muted"])))
        self._refresh_event(self._last_payload.get("time", 0.0)
                            if isinstance(self._last_payload, dict) else 0.0)

    def _label(self, text="", object_name=None, align=None):
        label = self.QtWidgets.QLabel(text)
        if object_name:
            label.setObjectName(object_name)
        if align is not None:
            label.setAlignment(align)
        return label

    def _card(self, object_name):
        frame = self.QtWidgets.QFrame()
        frame.setObjectName(object_name)
        layout = self.QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        return frame, layout

    def _legend_swatch(self, color):
        swatch = self.QtWidgets.QLabel()
        swatch.setFixedSize(12, 3)
        swatch.setStyleSheet(
            "QLabel {{ background-color: {}; border-radius: 1px; }}".format(color)
        )
        return swatch

    def _plot_card(self, title, unit, series, key, caption=None):
        W = self.QtWidgets
        card, layout = self._card("plotCard")
        header = W.QHBoxLayout()
        header.addWidget(self._label(title + (f" ({unit})" if unit else ""), "cardTitle"))
        header.addStretch()
        expand = W.QPushButton("Expand")
        header.addWidget(expand)
        layout.addLayout(header)
        if len(series) > 1:
            legend = W.QHBoxLayout()
            for name, color, *rest in series:
                if rest and rest[0]:
                    continue
                legend.addWidget(self._legend_swatch(color))
                legend.addWidget(self._label(name, "kvKey"))
            legend.addStretch()
            layout.addLayout(legend)
        if caption:
            layout.addWidget(self._wrapping_label(caption))
        spec = {"title": title, "unit": unit, "series": series, "key": key,
                "values": np.empty((0, len(series))), "y_range": None, "caption": caption}
        plot, curves, cursor = self._make_plot(spec)
        spec.update(plot=plot, curves=curves, cursor=cursor)
        readout = self._wrapping_label("No samples yet")
        spec["readout"] = readout
        layout.addWidget(plot)
        layout.addWidget(readout)
        if self._plots:
            plot.setXLink(self._plots[0]["plot"])
        self._plots.append(spec)
        expand.clicked.connect(lambda: self._expand_plot(spec))
        return card, plot, tuple(curves)

    def _make_plot(self, spec, expanded=False):
        from pyqtgraph.Qt import QtGui

        plot = self.pg.PlotWidget()
        plot.setMinimumHeight(130)
        if not expanded:
            plot.setFixedHeight(140)
        plot.setSizePolicy(self.QtWidgets.QSizePolicy.Policy.Expanding, self.QtWidgets.QSizePolicy.Policy.Fixed)
        plot.setBackground(THEME["surface"])
        plot.hideButtons()
        plot.disableAutoRange()
        plot.setXRange(0, 10, padding=0)
        font = QtGui.QFont()
        font.setPixelSize(12)
        for name in ("bottom", "left"):
            axis = plot.getAxis(name)
            axis.setPen(self.pg.mkPen(THEME["border_strong"]))
            axis.setTextPen(self.pg.mkPen(THEME["muted"]))
            axis.setStyle(tickFont=font, tickLength=-4)
        plot.getAxis("left").setWidth(52)
        plot.showGrid(x=True, y=True, alpha=0.12)
        curves = []
        for name, color, *rest in spec["series"]:
            dashed = bool(rest and rest[0])
            curve = plot.plot(name=name, pen=self.pg.mkPen(color, width=2,
                style=self.QtCore.Qt.PenStyle.DashLine if dashed else self.QtCore.Qt.PenStyle.SolidLine))
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            curves.append(curve)
        self._add_event_marker(plot)
        cursor = self.pg.InfiniteLine(angle=90, movable=False, pen=self.pg.mkPen(THEME["text"], width=1))
        plot.addItem(cursor, ignoreBounds=True)
        plot.getViewBox().sigRangeChangedManually.connect(lambda axes: self._manual_range(axes, plot))
        plot.scene().sigMouseMoved.connect(lambda pos: self._cursor_hover(plot, pos))
        return plot, curves, cursor

    def _plot_mode_changed(self, mode):
        if mode == "Live 10 s":
            self._cursor_follow = True
        self._render_plots()

    def _manual_range(self, axes, source=None):
        if self._setting_ranges:
            return
        self.plot_mode.blockSignals(True)
        self.auto_y.blockSignals(True)
        if axes[0]:
            self.plot_mode.setCurrentText("Inspect")
        if axes[1]:
            self.auto_y.setChecked(False)
        self.plot_mode.blockSignals(False)
        self.auto_y.blockSignals(False)
        for dialog, spec, plot, curves, cursor in self._expanded:
            if plot is source:
                ranges = plot.viewRange()
                spec["plot"].setRange(xRange=ranges[0] if axes[0] else None,
                                      yRange=ranges[1] if axes[1] else None, padding=0)
                break
        self._render_plots()

    def _cursor_hover(self, plot, pos):
        if plot.sceneBoundingRect().contains(pos):
            x = plot.getViewBox().mapSceneToView(pos).x()
            self.cursor_time.setValue(x)

    def _cursor_changed(self, value):
        self._cursor_follow = False
        self._update_cursor(value)

    def _move_cursor(self, direction):
        if not self.time_history:
            return
        times = np.asarray(self.time_history)
        index = int(np.argmin(np.abs(times - self.cursor_time.value())))
        index = int(np.clip(index + direction, 0, len(times) - 1))
        self.cursor_time.setValue(float(times[index]))

    def _update_cursor(self, value):
        if not self.time_history:
            return
        times = np.asarray(self.time_history)
        index = int(np.argmin(np.abs(times - value)))
        for spec in self._plots:
            spec["cursor"].setPos(times[index])
            numbers = spec["values"]
            if len(numbers) <= index:
                continue
            parts = []
            for (name, *_), number in zip(spec["series"], numbers[index]):
                parts.append(f"{name}: {number:.3f}" if np.isfinite(number) else f"{name}: Not available")
            spec["readout"].setText(f"{times[index]:.3f} s · " + "; ".join(parts) + (" " + spec["unit"] if spec["unit"] else ""))
        for dialog, spec, plot, curves, cursor in self._expanded:
            cursor.setPos(times[index])
            dialog.readout.setText(spec["readout"].text())

    def _render_plots(self, fit=False):
        if not self._plots:
            return
        times = np.asarray(self.time_history)
        mode = self.plot_mode.currentText()
        self._setting_ranges = True
        try:
            if mode != "Inspect":
                end = max(10.0 if mode == "Live 10 s" else 0.1, float(times[-1]) if times.size else 0.0)
                start = max(0.0, end - 10.0) if mode == "Live 10 s" else 0.0
                self._plots[0]["plot"].setXRange(start, end, padding=0)
            start, end = self._plots[0]["plot"].viewRange()[0]
            mask = (times >= start) & (times <= end)
            for spec in self._plots:
                values = spec["values"]
                for curve, column in zip(spec["curves"], values.T):
                    curve.setData(times, column)
                if self.auto_y.isChecked() or fit:
                    finite = values[mask] if len(values) == len(times) else np.empty(0)
                    finite = finite[np.isfinite(finite)]
                    if spec["key"] == "rotor" and self.rotor_plot_kind == "effectiveness":
                        lo, hi = -0.05, 1.05
                    elif finite.size:
                        lo, hi = min(0.0, float(finite.min())), max(0.05, float(finite.max()))
                        padding = max(0.01, (hi - lo) * 0.12)
                        lo, hi = lo - padding if lo < 0 else 0.0, hi + padding
                    else:
                        lo, hi = 0.0, 0.05
                    old = spec["y_range"]
                    if old and not fit:
                        lo, hi = min(lo, old[0]), max(hi, old[1])
                    spec["y_range"] = (lo, hi)
                    spec["plot"].setYRange(lo, hi, padding=0)
            for dialog, spec, plot, curves, cursor in self._expanded:
                for curve, column in zip(curves, spec["values"].T):
                    curve.setData(times, column)
                plot.setRange(xRange=spec["plot"].viewRange()[0], yRange=spec["plot"].viewRange()[1], padding=0)
        finally:
            self._setting_ranges = False
        if times.size:
            self.cursor_time.blockSignals(True)
            self.cursor_time.setMaximum(float(times[-1]))
            if self._cursor_follow:
                self.cursor_time.setValue(float(times[-1]))
            self.cursor_time.blockSignals(False)
            self._update_cursor(self.cursor_time.value())

    def _expand_plot(self, spec):
        dialog = self.QtWidgets.QDialog(self.window)
        dialog.setAttribute(self.QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(spec["title"])
        dialog.resize(960, 560)
        layout = self.QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(self._wrapping_label(spec["title"] + (" (" + spec["unit"] + ")" if spec["unit"] else "")))
        if spec["caption"]:
            layout.addWidget(self._wrapping_label(spec["caption"]))
        plot, curves, cursor = self._make_plot(spec, expanded=True)
        plot.setSizePolicy(self.QtWidgets.QSizePolicy.Policy.Expanding, self.QtWidgets.QSizePolicy.Policy.Expanding)
        plot.setLabel("bottom", "Time", units="s")
        if len(curves) > 1:
            legend = plot.addLegend(labelTextColor=THEME["text"], labelTextSize="12px")
            for curve, series in zip(curves, spec["series"]):
                legend.addItem(curve, series[0])
        # Copy ranges explicitly: ViewBox links align screen pixels, which
        # changes the source range when the expanded window is wider.
        layout.addWidget(plot)
        dialog.readout = self._wrapping_label("No samples yet")
        layout.addWidget(dialog.readout)
        navigation = self.QtWidgets.QHBoxLayout()
        for label, direction in (("Previous sample", -1), ("Next sample", 1)):
            button = self.QtWidgets.QPushButton(label)
            button.clicked.connect(lambda checked=False, step=direction: self._move_cursor(step))
            navigation.addWidget(button)
        layout.addLayout(navigation)
        entry = (dialog, spec, plot, curves, cursor)
        self._expanded.append(entry)
        dialog.finished.connect(lambda: self._expanded.remove(entry) if entry in self._expanded else None)
        self._render_plots()
        dialog.show()

    def close_expanded(self):
        for dialog, *_ in list(self._expanded):
            dialog.close()

    def _add_event_marker(self, plot):
        """Vertical dashed line at the wind/fault onset, shared by all plots."""
        if self.event_time_s is None or not np.isfinite(self.event_time_s):
            return
        marker = self.pg.InfiniteLine(
            pos=self.event_time_s,
            angle=90,
            movable=False,
            pen=self.pg.mkPen(
                THEME["faint"],
                width=1,
                style=self.QtCore.Qt.PenStyle.DashLine,
            ),
            label=self.event_label or None,
            labelOpts=(
                {
                    "position": 0.72,
                    "color": THEME["muted"],
                    "anchors": ((0, 1), (1, 1)),
                }
                if self.event_label
                else None
            ),
        )
        plot.addItem(marker)

    def _scene_center_and_extent(self):
        if self.waypoints.size == 0:
            return np.array([0.0, 0.0, 1.0], dtype=float), 3.0

        min_pt = np.min(self.waypoints, axis=0)
        max_pt = np.max(self.waypoints, axis=0)
        min_pt[2] = min(0.0, min_pt[2])
        center = 0.5 * (min_pt + max_pt)
        extent = max(float(np.max(max_pt - min_pt)), 1.0)
        return center, extent

    def _add_mesh(self, verts, faces, color, smooth=True, gl_options="opaque"):
        item = self.gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            smooth=smooth,
            color=color,
            shader="shaded",
            glOptions=gl_options,
        )
        self.view.addItem(item)
        return item

    def _build_body(self):
        """Rigid-body mesh set in body frame; transformed per frame in _set_frame."""
        xy = self._XY
        components = []

        def add(verts, faces, color, smooth=True, gl_options="opaque",
                spin_center=None, spin_dir=0.0):
            verts = verts * self.BODY_SCALE
            if spin_center is not None:
                spin_center = np.asarray(spin_center, dtype=float) * self.BODY_SCALE
            item = self._add_mesh(verts, faces, color, smooth, gl_options)
            components.append(
                {
                    "item": item,
                    "verts": verts,
                    "faces": faces,
                    "spin_center": spin_center,
                    "spin_dir": spin_dir,
                }
            )

        # Central hub + two diagonal arms (carbon frame).
        v, f = _box_mesh(0.034, 0.034, 0.009, center=(0, 0, 0.002))
        add(v, f, (0.78, 0.82, 0.88, 1.0), smooth=False)
        v, f = _box_mesh(self._ARM_LEN, 0.006, 0.0035, rot_z_deg=45.0)
        add(v, f, (0.58, 0.63, 0.71, 1.0), smooth=False)
        v, f = _box_mesh(self._ARM_LEN, 0.006, 0.0035, rot_z_deg=-45.0)
        add(v, f, (0.58, 0.63, 0.71, 1.0), smooth=False)

        # Real Crazyflie 2.x layout (+x forward, +y left): m1 front-right,
        # m2 rear-right, m3 rear-left, m4 front-left (model/params.py
        # body_frame order).
        motor_positions = [(xy, -xy), (-xy, -xy), (-xy, xy), (xy, xy)]
        cap_colors = [
            _gl_color("#ff8a3d"),  # m1/m4 are +x (front): warm accent
            _gl_color("#5aa2ff"),  # m2/m3 rear: cool
            _gl_color("#5aa2ff"),
            _gl_color("#ff8a3d"),
        ]
        prop_colors = [
            _gl_color("#ff8a3d", 0.40),
            _gl_color("#5aa2ff", 0.40),
            _gl_color("#5aa2ff", 0.40),
            _gl_color("#ff8a3d", 0.40),
        ]
        # body_frame order: m1(+x,-y) m2(-x,-y) m3(-x,+y) m4(+x,+y)
        front = (True, False, False, True)
        for i, (mx, my) in enumerate(motor_positions):
            v, f = _cylinder_mesh(
                self._MOTOR_R, self._MOTOR_H, center=(mx, my, self._MOTOR_H / 2.0)
            )
            add(v, f, (0.30, 0.34, 0.41, 1.0))
            v, f = _cylinder_mesh(
                self._MOTOR_R * 1.02, 0.0018, center=(mx, my, self._MOTOR_H + 0.001)
            )
            add(
                v, f,
                cap_colors[i] if front[i] else cap_colors[i],
                smooth=True,
            )
            v, f = _disc_mesh(self._PROP_R, center=(mx, my, self._PROP_Z))
            add(
                v, f,
                prop_colors[i],
                smooth=False,
                gl_options="translucent",
                spin_center=np.array([mx, my, self._PROP_Z]),
                spin_dir=self._PROP_DIRS[i],
            )

        return components

    def _init_scene(self):
        center, extent = self._scene_center_and_extent()
        grid_size = max(2.0, 1.4 * extent)
        distance = max(2.6, 2.0 * extent)

        self.view.setCameraPosition(distance=distance, elevation=24.0, azimuth=-55.0)
        try:
            self.view.opts["center"] = self.pg.Vector(
                float(center[0]), float(center[1]), float(center[2])
            )
        except Exception:
            pass

        # Two-layer ground grid: fine minor + coarse major.  Built from line
        # segments (GLGridItem's immediate-mode calls render nothing here).
        for spacing, color, width in (
            (0.25, self.COLOR_GRID_MINOR, 1.0),
            (1.0, self.COLOR_GRID_MAJOR, 1.0),
        ):
            self.view.addItem(
                self.gl.GLLinePlotItem(
                    glOptions="opaque",
                    pos=_grid_lines(grid_size / 2.0, spacing),
                    color=color,
                    width=width,
                    antialias=True,
                    mode="lines",
                )
            )

        # Long world axes are an optional reference, hidden in the clean view.
        axis_len = 0.35 * grid_size
        axis_defs = (
            (np.array([axis_len, 0.0, 0.0]), _gl_mix("#f2555a", THEME["scene"], 0.75), "X"),
            (np.array([0.0, axis_len, 0.0]), _gl_mix("#3dd68c", THEME["scene"], 0.75), "Y"),
            (np.array([0.0, 0.0, axis_len]), _gl_mix("#5aa2ff", THEME["scene"], 0.75), "Z"),
        )
        self.world_axes = []
        for tip, color, _letter in axis_defs:
            axis = self.gl.GLLinePlotItem(
                glOptions="opaque",
                # Lifted 1 mm so the axes do not z-fight with the grid.
                pos=np.vstack((np.zeros(3), tip)) + np.array([0.0, 0.0, 0.001]),
                color=color,
                width=2,
                antialias=True,
            )
            self.view.addItem(axis)
            axis.setVisible(False)
            self.world_axes.append(axis)
        self.origin_marker = self.gl.GLLinePlotItem(
            glOptions="opaque", mode="lines", width=2, antialias=True,
            pos=np.array([[-0.025, 0, 0.002], [0.025, 0, 0.002],
                          [0, -0.025, 0.002], [0, 0.025, 0.002]]),
            color=_gl_mix(THEME["muted"], THEME["scene"], 0.55),
        )
        self.view.addItem(self.origin_marker)
        # Axis labels: small gray overlay labels (real font) projected to
        # screen space each tick; GLTextItem renders nothing in this
        # Qt6/OpenGL context, so plain QLabels on top of the GL view it is.
        self.axis_label_tips = [(tip * 1.06, letter) for tip, _c, letter in axis_defs]
        self.axis_labels = []
        for _tip, letter in self.axis_label_tips:
            label = self.QtWidgets.QLabel(letter.lower(), self.view)
            label.setStyleSheet(
                "QLabel {{ color: {}; font-size: 12px; font-weight: 600;"
                " background: transparent; }}".format(THEME["muted"])
            )
            label.setAttribute(
                self.QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
            )
            label.hide()
            self.axis_labels.append(label)
        self.world_indicator = _make_world_indicator(self.view, self.QtCore, self.QtWidgets)
        self._axis_label_timer = self.QtCore.QTimer(self.window)
        self._axis_label_timer.timeout.connect(self._update_axis_labels)
        self._axis_label_timer.start(33)

        # Waypoints: sphere markers (GLScatterPlotItem renders white in this
        # Qt6/OpenGL context, so markers are meshes).  Skipped when every
        # waypoint is the same point (hover modes repeat the setpoint), so
        # no marker ball sits on top of the hovering quad.
        show_waypoints = self.waypoints.size and not np.allclose(
            self.waypoints, self.waypoints[0]
        )
        if show_waypoints:
            sphere_verts, sphere_faces = [], []
            vert_offset = 0
            for wp in self.waypoints:
                v, f = _sphere_mesh(0.014, center=wp)
                sphere_verts.append(v)
                sphere_faces.append(f + vert_offset)
                vert_offset += len(v)
            self._add_mesh(
                np.vstack(sphere_verts),
                np.vstack(sphere_faces),
                _gl_color("#ffc04d"),  # shaded lighting darkens; start bright
                smooth=True,
            )
            halo_verts, halo_faces = [], []
            vert_offset = 0
            for wp in self.waypoints:
                v, f = _sphere_mesh(0.026, center=wp, segments=12, rings=8)
                halo_verts.append(v)
                halo_faces.append(f + vert_offset)
                vert_offset += len(v)
            self._add_mesh(
                np.vstack(halo_verts),
                np.vstack(halo_faces),
                _gl_color("#f5a524", 0.22),
                smooth=True,
                gl_options="translucent",
            )

        # Quad body meshes + thrust-axis tick (body +z; the last world_frame
        # point sits above the centre, not ahead of it).
        self.body_components = self._build_body()
        self.thrust_axis_line = self.gl.GLLinePlotItem(
            glOptions="translucent",
            pos=np.zeros((2, 3), dtype=float),
            color=_gl_color(THEME["text"]),
            width=3,
            antialias=True,
        )
        self.view.addItem(self.thrust_axis_line)
        # An asymmetric nose arrow makes body yaw visible even when the
        # cross-shaped airframe and vertical thrust axis look stationary.
        self.nose_arrow_body = np.array([
            [0.0, 0.0, 0.014], [0.075, 0.0, 0.014],
            [0.075, 0.0, 0.014], [0.055, 0.012, 0.014],
            [0.075, 0.0, 0.014], [0.055, -0.012, 0.014],
        ]) * self.BODY_SCALE
        self.nose_arrow = self.gl.GLLinePlotItem(
            glOptions="opaque", pos=self.nose_arrow_body.copy(),
            color=_gl_color("#ffc04d"), width=3, antialias=True, mode="lines",
        )
        self.view.addItem(self.nose_arrow)

        # Flown trajectory: one dashed line (dashes cut from the history
        # polyline in _update_history).
        self.trail_line = self.gl.GLLinePlotItem(
            glOptions="translucent",
            pos=np.empty((0, 3), dtype=float),
            color=self.COLOR_TRAIL,
            width=2.0,
            antialias=True,
            mode="lines",
        )
        self.view.addItem(self.trail_line)

        # Optional faint, dashed altitude projection; independent of the trail.
        self.tether_line = self.gl.GLLinePlotItem(
            glOptions="opaque",
            pos=np.empty((0, 3), dtype=float),
            color=_gl_mix(THEME["muted"], THEME["scene"], 0.35),
            width=1,
            antialias=True,
            mode="lines",
        )
        self.view.addItem(self.tether_line)
        self.tether_line.setVisible(False)
        self.scene_options = self.QtWidgets.QToolButton(self.view)
        self.scene_options.setText("View")
        self.scene_options.setAccessibleName("Scene display options")
        self.scene_options.setToolTip("Show world axes or the dashed height projection")
        self.scene_options.setPopupMode(self.QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = self.QtWidgets.QMenu(self.scene_options)
        self.show_world_axes = menu.addAction("World axes")
        self.show_world_axes.setCheckable(True)
        self.show_world_axes.toggled.connect(self._set_world_axes_visible)
        self.show_height_projection = menu.addAction("Height projection")
        self.show_height_projection.setCheckable(True)
        self.show_height_projection.toggled.connect(self.tether_line.setVisible)
        self.scene_options.setMenu(menu)
        self.scene_options.adjustSize()
        self.scene_options.move(14, 14)
        self.scene_options.show()
        if self.show_wind:
            self.wind_arrow = self.gl.GLLinePlotItem(
                    glOptions="translucent",
                pos=np.empty((0, 3), dtype=float),
                color=_gl_color(self.COLOR_WIND_ARROW),
                width=3,
                antialias=True,
            )
            self.view.addItem(self.wind_arrow)
        else:
            self.wind_arrow = None

        # Key to the scene's marks and how to move the camera, as overlay
        # labels like the axis letters.
        entries = [("#5cc8ff", "flown path"), ("#ffc04d", "nose (body +X)"), (THEME["text"], "thrust axis"),
                   ("#ff8a3d", "front rotors")]
        if self.show_wind:
            entries.append((self.COLOR_WIND_ARROW, "wind velocity, 0.5 m per m/s" if self.wind_as_velocity else "wind force, 0.5 m per N"))
        if show_waypoints:
            entries.append(("#ffc04d", "waypoints"))
        self.scene_legend = self._overlay_label(
            "&nbsp;&nbsp;&nbsp;".join(
                "<span style='color:{}'>&#9644;</span>&nbsp;{}".format(color, text)
                for color, text in entries
            )
        )
        self._scene_legend_width = self.scene_legend.width()
        self.scene_legend.setWordWrap(True)
        self.scene_hint = self._overlay_label("Drag to orbit · scroll to zoom")
        self._update_axis_labels()

    def _set_world_axes_visible(self, visible):
        for axis in self.world_axes:
            axis.setVisible(visible)
        self._update_axis_labels()

    def _overlay_label(self, text):
        label = self.QtWidgets.QLabel(text, self.view)
        label.setStyleSheet(
            "QLabel {{ color: {muted}; background-color: rgba(11, 16, 24, 190);"
            " border: 1px solid {border}; border-radius: 8px; padding: 5px 10px;"
            " font-size: 12px; }}".format(**THEME)
        )
        label.setAttribute(self.QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        label.adjustSize()
        label.show()
        return label

    def start(self):
        # Place the body at the first simulated pose before the window is
        # painted; otherwise the body-frame mesh flashes at the world origin.
        self._on_timer()
        self.window.show()
        self.timer.start()

    def _on_timer(self):
        if self.frame_index >= self.frames:
            self.timer.stop()
            self._run_complete = True
            self.refresh_last_frame()
            self.set_run_state("Done")
            return

        payload = self.get_world_frame(self.frame_index)
        self.frame_index += 1
        self._set_frame(payload)

    def _set_frame(self, payload, *, force=False, record=True):
        (
            frame,
            sim_time,
            wind_force,
            yaw_rate_dps,
            position_error_norm,
            position_error_norm_for_max,
            position_error_max_payload,
            waiting_for_wind,
            position_error_waiting_label,
            controller_mode,
            rotor_fault_case,
            rotor_fault_rotor,
            rotor_fault_start_time,
            rotor_fault_state,
            attitude_deg,
            controller_status_label,
            controller_status_value,
            motor_thrusts,
            effectiveness_truth,
            effectiveness_estimate,
        ) = _parse_payload(payload)

        if frame.shape != (3, 6):
            raise ValueError("Frame must be a 3x6 matrix.")
        self._last_payload = payload

        rot, origin = _frame_rotation_and_origin(frame)
        self._update_body(rot, origin, frame, animate=record)
        self._update_wind_arrow(origin, wind_force)
        if record:
            self._update_history(origin, sim_time)

        # Side-panel repaint is the main per-frame UI cost (antialiased
        # polylines + axis relayout on setXRange/setYRange).  Refresh it at
        # half rate so mouse drags on the 3D view are not starved. Retain
        # every display sample even when its repaint is deferred.
        self._ui_phase = (self._ui_phase + 1) % 2
        self._update_telemetry(
            sim_time,
            wind_force,
            yaw_rate_dps,
            position_error_norm,
            motor_thrusts,
            effectiveness_truth,
            effectiveness_estimate,
            record=record,
            paint=force or not self._ui_phase,
        )
        if self._ui_phase and not force:
            return
        self._update_status(
            frame,
            origin,
            sim_time,
            wind_force,
            yaw_rate_dps,
            position_error_norm,
            position_error_norm_for_max,
            position_error_max_payload,
            waiting_for_wind,
            position_error_waiting_label,
            controller_mode,
            rotor_fault_case,
            rotor_fault_rotor,
            rotor_fault_start_time,
            rotor_fault_state,
            attitude_deg,
            controller_status_label,
            controller_status_value,
        )

    def refresh_last_frame(self):
        if self._last_payload is not None:
            self._set_frame(self._last_payload, force=True, record=False)

    def _update_body(self, rot, origin, frame, animate=True):
        if animate:
            self.prop_angle += np.radians(420.0) / self.fps
        for comp in self.body_components:
            verts = comp["verts"]
            if comp["spin_center"] is not None:
                verts = _spin_verts(
                    verts, comp["spin_center"], self.prop_angle * comp["spin_dir"]
                )
            comp["item"].setMeshData(
                vertexes=verts @ rot.T + origin, faces=comp["faces"]
            )
        thrust_tip = frame[:, 4] + (frame[:, 5] - frame[:, 4]) * self.BODY_SCALE
        self.thrust_axis_line.setData(antialias=True, pos=np.column_stack((frame[:, 4], thrust_tip)).T)
        self.nose_arrow.setData(antialias=True, pos=self.nose_arrow_body @ rot.T + origin)
        ground = np.array([origin[0], origin[1], 0.0])
        self.tether_line.setData(antialias=True, pos=_dashify(np.vstack((ground, origin)), 0.035, 0.035))

    def _update_wind_arrow(self, origin, wind_force):
        if self.wind_arrow is None:
            return
        norm = float(np.linalg.norm(wind_force))
        if norm < 1e-4:
            self.wind_arrow.setData(antialias=True, pos=np.empty((0, 3), dtype=float))
            return
        tip = origin + 0.5 * wind_force  # 0.5 m per Newton
        self.wind_arrow.setData(antialias=True, pos=np.vstack((origin, tip)))

    def _update_axis_labels(self):
        """Update camera orientation and position the scene overlays."""
        from pyqtgraph.Qt import QtGui

        view = self.view
        w, h = view.width(), view.height()
        if w <= 0 or h <= 0:
            return
        mat = view.viewMatrix()
        axes = [mat.mapVector(QtGui.QVector3D(*axis)) for axis in np.eye(3)]
        self.world_indicator.set_axes([[axis.x(), axis.y(), axis.z()] for axis in axes])
        self.world_indicator.move(w - self.world_indicator.width() - 14, 14)
        viewport = (
            view.getViewport()
            if hasattr(view, "getViewport")
            else (0, 0, w, h)
        )
        try:
            proj = view.projectionMatrix(viewport, viewport)
        except TypeError:
            proj = view.projectionMatrix()
        except Exception:
            return
        for (tip, _letter), label in zip(self.axis_label_tips, self.axis_labels):
            if not self.show_world_axes.isChecked():
                label.hide()
                continue
            world = QtGui.QVector4D(float(tip[0]), float(tip[1]), float(tip[2]), 1.0)
            clip = proj.map(mat.map(world))
            if clip.w() <= 1e-6:
                label.hide()
                continue
            ndc = clip.toVector3D() / clip.w()
            x = (ndc.x() + 1.0) * 0.5 * w
            y = (1.0 - ndc.y()) * 0.5 * h
            label.move(int(x - label.width() / 2), int(y - label.height() / 2))
            label.show()
        legend_width = min(self._scene_legend_width, max(100, w - 28))
        if self.scene_legend.width() != legend_width:
            self.scene_legend.setFixedWidth(legend_width)
        self.scene_legend.setFixedHeight(self.scene_legend.heightForWidth(legend_width))
        self.scene_legend.move(14, h - self.scene_legend.height() - 14)
        hint_bottom = 14
        if legend_width + self.scene_hint.width() + 42 > w:
            hint_bottom += self.scene_legend.height() + 8
        self.scene_hint.move(max(14, w - self.scene_hint.width() - 14), h - self.scene_hint.height() - hint_bottom)

    def _update_history(self, origin, sim_time):
        self.history.append(origin)
        self.history_time.append(sim_time)
        if self.history_window_s is not None and np.isfinite(sim_time):
            while (
                self.history_time
                and np.isfinite(self.history_time[0])
                and (sim_time - self.history_time[0]) > self.history_window_s
            ):
                self.history.popleft()
                self.history_time.popleft()

        if not self.history:
            return
        points = np.asarray(self.history, dtype=float)
        self.trail_line.setData(
            antialias=True,
            pos=_dashify(points, self.TRAIL_DASH_M, self.TRAIL_GAP_M),
        )

    def _update_telemetry(self, sim_time, wind_force, yaw_rate_dps, position_error_norm,
                          motor_thrusts, effectiveness_truth, effectiveness_estimate, *, record=True, paint=True):
        if not np.isfinite(sim_time):
            return
        if record and (not self.time_history or sim_time > self.time_history[-1]):
            self.time_history.append(sim_time)
            self.wind_force_history.append(wind_force.copy())
            self.yaw_rate_history.append(yaw_rate_dps)
            self.pos_err_history.append(float(position_error_norm))
            for history, values in ((self.motor_thrust_history, motor_thrusts),
                                    (self.eff_est_history, effectiveness_estimate),
                                    (self.eff_truth_history, effectiveness_truth)):
                history.append(np.full(4, np.nan) if values is None else values.copy())
        if not paint:
            return
        n = len(self.time_history)
        data = {"error": np.asarray(self.pos_err_history).reshape(n, 1),
                "spin": np.asarray(self.yaw_rate_history).reshape(n, 1),
                "wind": np.asarray(self.wind_force_history).reshape(n, 3)}
        data["rotor"] = (np.column_stack((np.asarray(self.eff_est_history).reshape(n, 4),
                                            np.asarray(self.eff_truth_history).reshape(n, 4)))
                         if self.rotor_plot_kind == "effectiveness"
                         else np.asarray(self.motor_thrust_history).reshape(n, 4))
        for spec in self._plots:
            spec["values"] = data[spec["key"]]
        self._render_plots()

    def _update_status(
        self,
        frame,
        origin,
        sim_time,
        wind_force,
        yaw_rate_dps,
        position_error_norm,
        position_error_norm_for_max,
        position_error_max_payload,
        waiting_for_wind,
        position_error_waiting_label,
        controller_mode,
        rotor_fault_case,
        rotor_fault_rotor,
        rotor_fault_start_time,
        rotor_fault_state,
        attitude_deg,
        controller_status_label,
        controller_status_value,
    ):
        if np.all(np.isfinite(attitude_deg)):
            # Plant-integrated attitude: exact and unwrapped (yaw keeps
            # accumulating past +/-180 deg in yaw-released spinning flight).
            roll_deg, pitch_deg, yaw_deg = (float(a) for a in attitude_deg)
        else:
            # Raw-frame payloads carry no attitude; fall back to the
            # matrix-derived readout (yaw folds to +/-180 deg).
            roll_deg, pitch_deg, yaw_deg = _frame_attitude_degrees(frame)

        if np.isfinite(position_error_max_payload):
            self.position_error_max = position_error_max_payload
        elif np.isfinite(position_error_norm_for_max):
            if not np.isfinite(self.position_error_max):
                self.position_error_max = position_error_norm_for_max
            else:
                self.position_error_max = max(
                    self.position_error_max, position_error_norm_for_max
                )

        self.brand.setText("Crazyflie Simulator · " + _pretty_controller(controller_mode))
        diagnostic = _format_status(controller_status_label, controller_status_value)
        if controller_status_label and "lambda" in controller_status_label and np.isfinite(controller_status_value):
            diagnostic = f"Minimum allocated effectiveness: {controller_status_value:.3f}\nRaw value: {controller_status_value!r}"
        self.controller_status.setText(diagnostic if controller_status_label else "Not available")
        if np.isfinite(sim_time):
            self.time_value.setText(f"{sim_time:.2f} / {self.duration_s:.2f} s")
            self.progress.setValue(int(np.clip(sim_time / max(self.duration_s, 1e-9), 0, 1) * 1000))
        else:
            self.time_value.setText("—")

        self._set_xyz("pos", origin, "+.2f")
        self._set_xyz("att", (roll_deg, pitch_deg, yaw_deg), "+.1f")
        if self.show_wind:
            self._set_xyz("wind", wind_force, "+.3f")

        yaw_rate_text = _fmt(yaw_rate_dps, "+.1f")
        self._stat_values["yaw_rate"].setText(
            "Not available" if yaw_rate_text == "—" else yaw_rate_text + " °/s"
        )
        if self.show_wind:
            wind_norm = float(np.linalg.norm(wind_force))
            wind_units = " m/s" if self.wind_as_velocity else " N"
            self._stat_values["wind_norm"].setText(
                _fmt(wind_norm, ".3f") + wind_units
            )

        if np.isfinite(position_error_norm):
            self._stat_values["pos_err"].setText(
                _fmt(position_error_norm, ".3f") + " m"
            )
            if waiting_for_wind:
                wait_label = position_error_waiting_label or "event"
                self._stat_values["pos_err_max"].setText("wait " + wait_label)
            else:
                max_text = _fmt(self.position_error_max, ".3f")
                self._stat_values["pos_err_max"].setText(
                    "Not available" if max_text == "—" else max_text + " m"
                )
        else:
            self._stat_values["pos_err"].setText("Not available")
            self._stat_values["pos_err_max"].setText("Not available")

        if rotor_fault_case and not self.event_description:
            rotors = self._last_payload.get("rotor_fault_rotors", [rotor_fault_rotor]) if isinstance(self._last_payload, dict) else [rotor_fault_rotor]
            from crazyflie_sim.sim.scenario import fault_case_text

            self.event_description = f"Rotor {', '.join(map(str, rotors))}: {fault_case_text(rotor_fault_case)}"
            if np.isfinite(rotor_fault_start_time):
                self.event_time_s = rotor_fault_start_time
        self._refresh_event(sim_time)

    def _set_xyz(self, key, values, spec):
        for label, value in zip(self._xyz_values[key], values):
            label.setText(_fmt(value, spec, fallback="Not available"))


def plot_quad_3d(
    waypoints,
    get_world_frame,
    frames=400,
    history_window_s=None,
    fps=50,
    view_config=None,
):
    pg, gl, QtCore, QtWidgets = _load_qt_modules()

    app = QtWidgets.QApplication.instance()
    created_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    window = _PyQtGraphQuadWindow(
        waypoints=waypoints,
        get_world_frame=get_world_frame,
        frames=frames,
        history_window_s=history_window_s,
        fps=fps,
        pg=pg,
        gl=gl,
        QtCore=QtCore,
        QtWidgets=QtWidgets,
        view_config=view_config,
    )
    window.start()

    if created_app:
        return _qt_exec(app)
    return window
