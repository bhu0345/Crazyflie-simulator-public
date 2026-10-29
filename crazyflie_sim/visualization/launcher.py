"""Interactive controls around the existing live view and Simulation loop."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml

from crazyflie_sim.controllers import registry
from crazyflie_sim.model.actuator_profile import available_actuator_profiles, get_actuator_profile
from crazyflie_sim.runsim import ANIMATION_HZ, HISTORY_WINDOW_S, make_frame_source, view_config
from crazyflie_sim.sim.config import from_dict, load_config, params_overridden
from crazyflie_sim.sim.scenario import fault_case_text
from crazyflie_sim.sim.simulation import Simulation
from crazyflie_sim.sim import archive
from crazyflie_sim.visualization.run_result import RunResult
from crazyflie_sim.visualization.quad_plot_pg import THEME, _load_qt_modules, _PyQtGraphQuadWindow, scroll_qss


pg, gl, QtCore, QtWidgets = _load_qt_modules()
from pyqtgraph.Qt import QtGui  # noqa: E402  (after _load_qt_modules picks the binding)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

_LAUNCHER_QSS = """
    QWidget#launcherRoot {{ background-color: {bg}; }}
    QFrame#toolbar {{
        background-color: {surface};
        border: 1px solid {border};
        border-radius: 12px;
    }}
    QLabel {{ color: {text}; font-size: 12px; }}
    QDialog {{ background-color: {surface}; }}
    QMenu {{ background: {surface_alt}; color: {text}; padding: 6px; }}
    QMenu::item {{ padding: 8px 20px; }}
    QMenu::item:selected {{ background: {accent_strong}; color: white; }}
    QMenu::item:disabled {{ color: {faint}; }}
    QLabel#brandTitle {{ font-size: 15px; font-weight: 700; }}
    QLabel#fieldCaption {{
        color: {faint}; font-size: 12px; font-weight: 700; letter-spacing: 0.8px;
    }}
    QLabel#statusMessage {{ color: {muted}; font-size: 12px; }}

    QComboBox {{
        background-color: {surface_alt};
        color: {text};
        border: 1px solid {border_strong};
        border-radius: 8px;
        padding: 6px 28px 6px 10px;
        min-height: 20px;
        font-size: 13px;
        combobox-popup: 0;
    }}
    QComboBox:hover {{ border-color: {faint}; }}
    QComboBox:focus {{ border-color: {accent}; }}
    QComboBox:disabled {{ color: {faint}; background-color: {surface}; border-color: {border}; }}
    QComboBox::drop-down {{
        subcontrol-origin: padding; subcontrol-position: center right;
        width: 24px; border: none;
    }}
    QComboBox::down-arrow {{ image: url("{chevron}"); width: 10px; height: 6px; }}
    QComboBox::down-arrow:disabled {{ image: url("{chevron_disabled}"); }}
    QComboBox QAbstractItemView {{
        background-color: {surface_alt};
        color: {text};
        border: 1px solid {border_strong};
        padding: 4px;
        outline: 0;
        selection-background-color: {accent_strong};
        selection-color: #ffffff;
    }}

    QPushButton {{
        background-color: {surface_alt};
        color: {text};
        border: 1px solid {border_strong};
        border-radius: 8px;
        padding: 7px 14px;
        font-size: 13px;
        font-weight: 600;
    }}
    QPushButton:hover {{ background-color: {hover}; border-color: {faint}; }}
    QPushButton:pressed {{ background-color: {border}; }}
    QPushButton:focus {{ border-color: {accent}; }}
    QPushButton:disabled {{
        color: {faint}; background-color: transparent; border-color: {border};
    }}
    QPushButton#primaryButton {{
        background-color: {accent_strong}; border-color: {accent_strong}; color: #ffffff;
    }}
    QPushButton#primaryButton:hover {{ background-color: #3b7cf5; border-color: #3b7cf5; }}
    QPushButton#primaryButton:disabled {{
        background-color: #18294a; border-color: #18294a; color: #6d7f99;
    }}
    QPushButton#ghostButton {{
        background-color: transparent; border-color: transparent; color: {muted};
    }}
    QPushButton#ghostButton:hover {{ color: {text}; background-color: {surface_alt}; }}
    QPushButton#ghostButton:disabled {{ color: {faint}; }}
"""


def _painted(size, draw):
    """A crisp (2x) transparent pixmap of ``size`` logical pixels."""
    ratio = 2.0
    pixmap = QtGui.QPixmap(int(size[0] * ratio), int(size[1] * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    draw(painter)
    painter.end()
    return pixmap


def _icon_pixmap(kind, color, s=16):
    """Line icons drawn with QPainter: no image files, no emoji glyphs."""
    color = QtGui.QColor(color)
    pen = QtGui.QPen(color, 1.7)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    no_pen = QtCore.Qt.PenStyle.NoPen
    no_brush = QtCore.Qt.BrushStyle.NoBrush

    def draw(p):
        if kind == "play":
            path = QtGui.QPainterPath()
            path.moveTo(s * 0.32, s * 0.20)
            path.lineTo(s * 0.80, s * 0.50)
            path.lineTo(s * 0.32, s * 0.80)
            path.closeSubpath()
            p.setPen(no_pen)
            p.setBrush(color)
            p.drawPath(path)
        elif kind == "pause":
            p.setPen(no_pen)
            p.setBrush(color)
            for x in (0.27, 0.58):
                p.drawRoundedRect(QtCore.QRectF(s * x, s * 0.22, s * 0.15, s * 0.56), 1.0, 1.0)
        elif kind == "reset":
            p.setPen(pen)
            p.setBrush(no_brush)
            p.drawArc(QtCore.QRectF(s * 0.20, s * 0.20, s * 0.60, s * 0.60), 100 * 16, 290 * 16)
            # Arrowhead at the arc's start (top), pointing clockwise.
            path = QtGui.QPainterPath()
            path.moveTo(s * 0.36, s * 0.10)
            path.lineTo(s * 0.52, s * 0.20)
            path.lineTo(s * 0.38, s * 0.33)
            p.drawPath(path)
        elif kind == "open":
            p.setPen(pen)
            p.setBrush(no_brush)
            path = QtGui.QPainterPath()
            path.moveTo(s * 0.14, s * 0.78)
            path.lineTo(s * 0.14, s * 0.24)
            path.lineTo(s * 0.40, s * 0.24)
            path.lineTo(s * 0.48, s * 0.34)
            path.lineTo(s * 0.86, s * 0.34)
            path.lineTo(s * 0.86, s * 0.78)
            path.closeSubpath()
            p.drawPath(path)
        elif kind == "save":
            p.setPen(pen)
            p.setBrush(no_brush)
            p.drawLine(QtCore.QPointF(s * 0.50, s * 0.14), QtCore.QPointF(s * 0.50, s * 0.60))
            p.drawPolyline(QtGui.QPolygonF([
                QtCore.QPointF(s * 0.32, s * 0.44),
                QtCore.QPointF(s * 0.50, s * 0.62),
                QtCore.QPointF(s * 0.68, s * 0.44),
            ]))
            p.drawPolyline(QtGui.QPolygonF([
                QtCore.QPointF(s * 0.16, s * 0.66),
                QtCore.QPointF(s * 0.16, s * 0.84),
                QtCore.QPointF(s * 0.84, s * 0.84),
                QtCore.QPointF(s * 0.84, s * 0.66),
            ]))

    return _painted((s, s), draw)


def _icon(kind, color, disabled_color=THEME["faint"]):
    icon = QtGui.QIcon()
    icon.addPixmap(_icon_pixmap(kind, color), QtGui.QIcon.Mode.Normal)
    icon.addPixmap(_icon_pixmap(kind, disabled_color), QtGui.QIcon.Mode.Disabled)
    return icon


def _brand_pixmap(s=30):
    """The app mark: an X-frame quadrotor with its four rotors."""
    accent = QtGui.QColor(THEME["accent"])

    def draw(p):
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(THEME["surface_alt"]))
        p.drawRoundedRect(QtCore.QRectF(0, 0, s, s), 8, 8)
        pen = QtGui.QPen(accent, 2.0)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QtCore.QPointF(s * 0.30, s * 0.30), QtCore.QPointF(s * 0.70, s * 0.70))
        p.drawLine(QtCore.QPointF(s * 0.70, s * 0.30), QtCore.QPointF(s * 0.30, s * 0.70))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for cx, cy in ((0.27, 0.27), (0.73, 0.27), (0.27, 0.73), (0.73, 0.73)):
            p.drawEllipse(QtCore.QPointF(s * cx, s * cy), s * 0.13, s * 0.13)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(accent)
        p.drawEllipse(QtCore.QPointF(s * 0.5, s * 0.5), s * 0.07, s * 0.07)

    return _painted((s, s), draw)


def _chevron_files():
    """Combo-box arrows for the stylesheet, which only takes image files.
    Qt picks the @2x file on high-DPI screens."""
    folder = Path(tempfile.gettempdir()) / "crazyflie_simulator_gui"
    folder.mkdir(exist_ok=True)
    paths = {}
    for name, color in (("chevron", THEME["muted"]), ("chevron_disabled", THEME["border_strong"])):
        for scale, suffix in ((1, ""), (2, "@2x")):
            pixmap = QtGui.QPixmap(10 * scale, 6 * scale)
            pixmap.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor(color), 1.5 * scale)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(QtGui.QPolygonF([
                QtCore.QPointF(1.0 * scale, 1.0 * scale),
                QtCore.QPointF(5.0 * scale, 5.0 * scale),
                QtCore.QPointF(9.0 * scale, 1.0 * scale),
            ]))
            painter.end()
            pixmap.save(str(folder / "{}{}.png".format(name, suffix)))
        paths[name] = (folder / "{}.png".format(name)).as_posix()
    return paths


def _profile_label(profile):
    name = profile.name.replace("crazyflie21", "Crazyflie 2.1").replace("_", " ")
    return f"{name} · {profile.mass * 1000:g} g"


def _scenario_summary(scenario, duration):
    event, _ = scenario.event()
    if scenario.sim_mode == "trajectory":
        goal = "Follow the reference trajectory"
        source = {"list": f"{len(scenario.user_waypoints)} waypoints",
                  "csv": "Waypoints from a CSV file",
                  "helix": "A helix"}.get(scenario.waypoint_source, "Waypoints")
        detail = f"{source} at {scenario.trajectory_speed:g} m/s"
    elif scenario.sim_mode == "hovering":
        goal, detail = "Hover in place", "No disturbances"
    elif scenario.sim_mode == "hover_wind":
        goal = "Hold position under wind"
        kind = "Wind velocity" if scenario.wind.as_velocity else "Wind force"
        detail = f"{kind} starts at {event:.3f} s"
    else:
        fault = scenario.active_fault()
        if fault.enabled:
            rotors = ", ".join(str(i) for i in fault.rotor_indices)
            goal = "Hold position after " + ("a single-rotor fault" if len(fault.rotor_indices) == 1 else "rotor faults")
            detail = f"Rotor {rotors}: {fault_case_text(fault.label)} at {event:.3f} s"
        else:
            goal, detail = "Hold position", "No rotor fault configured"
    return f"{goal}. {detail}. Duration {duration:.1f} s."


class SimulationLauncher(QtWidgets.QWidget):
    def __init__(self, config):
        super().__init__()
        self.viewer = None
        self.sim = None
        self.last_result = None
        self.playback_rate = 1.0
        self.scene_display = (False, False)
        self.result_dialog = None
        self.config = config
        self.setWindowTitle("Crazyflie Simulator")
        self.setObjectName("launcherRoot")
        self.setStyleSheet(_LAUNCHER_QSS.format(**THEME, **_chevron_files()) + scroll_qss())
        self.setMinimumSize(1024, 700)
        self.resize(1440, 900)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        toolbar = QtWidgets.QFrame()
        toolbar.setObjectName("toolbar")
        bar = QtWidgets.QVBoxLayout(toolbar)
        bar.setContentsMargins(16, 12, 16, 12)
        bar.setSpacing(10)
        commands = QtWidgets.QHBoxLayout()
        commands.setSpacing(10)
        mark = QtWidgets.QLabel()
        mark.setPixmap(_brand_pixmap())
        commands.addWidget(mark)
        title = QtWidgets.QLabel("Crazyflie Simulator")
        title.setObjectName("brandTitle")
        commands.addWidget(title)
        self.file_button = QtWidgets.QPushButton("File")
        menu = QtWidgets.QMenu(self.file_button)
        self.open_button = menu.addAction("Open config…", self.open_config)
        self.save_button = menu.addAction("Save config…", self.save_config)
        self.file_button.setMenu(menu)
        commands.addWidget(self.file_button)
        commands.addStretch(1)
        self.last_result_button = QtWidgets.QPushButton("Last result…")
        self.last_result_button.clicked.connect(self.show_last_result)
        self.last_result_button.setEnabled(False)
        commands.addWidget(self.last_result_button)
        self.reset_button = self._button("Reset", "reset", None, self.reset_run)
        self.start_button = self._button("Start", "play", "primaryButton", self.start_run, "#ffffff")
        self.pause_button = self._button("Pause", "pause", "primaryButton", self.pause_run, "#ffffff")
        self.reset_button.setToolTip("Prepare the initial state and restore random seeds (Ctrl+R)")
        self.start_button.setToolTip("Start or resume (Space)")
        self.pause_button.setToolTip("Pause (Space)")
        self.reset_button.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, self._toggle_run)
        for button in (self.reset_button, self.start_button, self.pause_button):
            commands.addWidget(button)
        bar.addLayout(commands)

        self.scenario_combo = QtWidgets.QComboBox()
        for path in sorted(CONFIGS_DIR.rglob("*")):
            if path.suffix.lower() in (".yaml", ".yml", ".json") and not path.name.startswith("_"):
                name = str(path.relative_to(CONFIGS_DIR).with_suffix("")).replace("_", " ")
                self.scenario_combo.addItem(name, str(path.resolve()))
                self.scenario_combo.setItemData(self.scenario_combo.count() - 1, str(path), QtCore.Qt.ItemDataRole.ToolTipRole)
        self.controller_combo = QtWidgets.QComboBox()
        for mode in registry.available():
            self.controller_combo.addItem(registry.get(mode).label, mode)
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.addItem("Default: " + _profile_label(get_actuator_profile()), None)
        for name in available_actuator_profiles():
            self.profile_combo.addItem(_profile_label(get_actuator_profile(name)), name)
        fields = QtWidgets.QHBoxLayout()
        fields.setSpacing(16)
        for label, combo, stretch in (
            ("Scenario", self.scenario_combo, 3),
            ("Controller", self.controller_combo, 2),
            ("Vehicle", self.profile_combo, 2),
        ):
            field = QtWidgets.QVBoxLayout()
            field.setSpacing(4)
            caption = QtWidgets.QLabel(label)
            caption.setObjectName("fieldCaption")
            caption.setBuddy(combo)
            combo.setAccessibleName(label)
            combo.setMinimumContentsLength(8)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            combo.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            QtGui.QShortcut(QtGui.QKeySequence("Alt+" + label[0]), self, combo.setFocus)
            field.addWidget(caption)
            field.addWidget(combo)
            fields.addLayout(field, stretch)
        bar.addLayout(fields)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(self.summary)
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("statusMessage")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(self.status)
        toolbar_row = QtWidgets.QHBoxLayout()
        toolbar_row.setContentsMargins(16, 16, 16, 0)
        toolbar_row.addWidget(toolbar)
        self.main_layout.addLayout(toolbar_row)
        self.scenario_combo.activated.connect(self._select_scenario)
        self.controller_combo.activated.connect(self.reset_run)
        self.profile_combo.activated.connect(self.reset_run)
        self._apply_config(config)

    def _button(self, text, icon, object_name, slot, icon_color=THEME["text"]):
        button = QtWidgets.QPushButton(text)
        if object_name:
            button.setObjectName(object_name)
        button.setIcon(_icon(icon, icon_color))
        button.setIconSize(QtCore.QSize(16, 16))
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(slot)
        return button

    def _toggle_run(self):
        if self.state == "Running":
            self.pause_run()
        else:
            self.start_run()

    def _apply_config(self, config):
        self.config = config
        if config.source:
            path = str(Path(config.source).resolve())
            index = self.scenario_combo.findData(path)
            if index < 0:
                self.scenario_combo.addItem(path, path)
                index = self.scenario_combo.count() - 1
            self.scenario_combo.setCurrentIndex(index)
            self.scenario_combo.setToolTip(path)
        self.controller_combo.setCurrentIndex(self.controller_combo.findData(config.controller_mode))
        self.profile_combo.setCurrentIndex(self.profile_combo.findData(config.scenario.actuator_profile))
        self.reset_run()

    def _select_scenario(self):
        previous = self.config.source
        try:
            config = load_config(self.scenario_combo.currentData())
        except Exception as exc:
            if previous:
                self.scenario_combo.setCurrentIndex(self.scenario_combo.findData(str(Path(previous).resolve())))
            self._show_error("Could not load config", exc)
            return
        self._apply_config(config)

    def open_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open run config", str(CONFIGS_DIR), "Run configs (*.yaml *.yml *.json)"
        )
        if path:
            try:
                config = load_config(path)
            except Exception as exc:
                self._show_error("Could not load config", exc)
                return
            self._apply_config(config)

    def selected_config(self):
        raw = self.config.to_dict()
        raw["controller_mode"] = self.controller_combo.currentData()
        raw["actuator_profile"] = self.profile_combo.currentData()
        return from_dict(raw, source=self.config.source)

    def _discard_view(self):
        if self.viewer is not None:
            self.playback_rate = self.viewer.playback_speed.currentData()
            self.scene_display = (self.viewer.show_world_axes.isChecked(),
                                  self.viewer.show_height_projection.isChecked())
            self.viewer.timer.stop()
            self.viewer._axis_label_timer.stop()
            self.viewer.close_expanded()
            self.main_layout.removeWidget(self.viewer.window)
            self.viewer.window.hide()
            self.viewer.window.deleteLater()
            self.viewer = None
        self.sim = None

    def reset_run(self):
        self._discard_view()
        try:
            self.run_config = self.selected_config()
            cfg = self.run_config
            # Scope even GUI ticks: switching scenarios must not inherit overrides.
            with params_overridden(cfg.params_overrides):
                self.sim = Simulation.from_scenario(cfg.scenario, cfg.controller_mode, cfg.gains())
                _, waypoints, _ = cfg.scenario.reference()
                self.viewer = _PyQtGraphQuadWindow(
                    waypoints, make_frame_source(self.sim, cfg.scenario, cfg.controller_mode),
                    int(np.ceil(self.sim.duration_s * ANIMATION_HZ)), HISTORY_WINDOW_S,
                    ANIMATION_HZ, pg, gl, QtCore, QtWidgets,
                    view_config=view_config(self.sim, cfg.scenario),
                )
            self.summary.setText(_scenario_summary(cfg.scenario, self.sim.duration_s))
            self.viewer.brand.setVisible(False)
            self.viewer.playback_speed.setCurrentIndex(
                self.viewer.playback_speed.findData(self.playback_rate))
            self.viewer.show_world_axes.setChecked(self.scene_display[0])
            self.viewer.show_height_projection.setChecked(self.scene_display[1])
            self.viewer.export_result.clicked.connect(self.export_results)
            details = (f"Config: {cfg.source or 'in-memory configuration'}\n"
                       f"Controller: {registry.get(cfg.controller_mode).label}\n"
                       f"Vehicle: {self.sim.quad.actuator_profile.name}\n"
                       f"Plant mass: {self.sim.quad.mass * 1000:g} g\n"
                       f"Plant overrides: {cfg.scenario.plant_overrides or 'None'}")
            self.viewer.configuration_details.setText(details)
            self.main_layout.addWidget(self.viewer.window, 1)
            self.viewer.timer.timeout.disconnect(self.viewer._on_timer)
            self.viewer.timer.timeout.connect(self._tick)
            # Preview the initial pose without stepping or sampling stochastic wind.
            self.viewer._ui_phase = 1
            self.viewer._set_frame({
                "frame": self.sim.quad.world_frame(), "time": 0.0,
                "attitude_deg": np.degrees(self.sim.quad.attitude()),
                "controller_mode": cfg.controller_mode,
                "position_error_norm": float(np.linalg.norm(
                    self.sim.quad.position() - self.sim.reference(0.0).pos)),
                "waiting_for_wind": cfg.scenario.event()[0] is not None,
            }, force=True, record=False)
        except Exception as exc:
            self._discard_view()
            self._set_state("Error", "Choose another configuration or reset to retry.")
            self._show_error("Could not prepare simulation", exc)
            return
        self._set_state("Ready", "Start to run. Reset returns to the initial state.")

    def _set_state(self, state, message):
        self.state = state
        editable = state in ("Ready", "Done", "Error")
        for widget in (self.scenario_combo, self.controller_combo, self.profile_combo, self.open_button):
            widget.setEnabled(editable)
        self.start_button.setVisible(state != "Running")
        self.start_button.setEnabled(state in ("Ready", "Paused", "Done"))
        self.start_button.setText({"Paused": "Resume", "Done": "Run again"}.get(state, "Start"))
        self.pause_button.setVisible(state == "Running")
        self.pause_button.setEnabled(state == "Running")
        # The live view's error card shows the state; without a view (a failed
        # prepare) the message says it.
        self.status.setText(message if self.viewer is not None else "{} · {}".format(state, message))
        if self.viewer is not None:
            self.viewer.set_run_state(state)

    def start_run(self):
        if self.state == "Done":
            self.reset_run()
        if self.state not in ("Ready", "Paused"):
            return
        self._set_state("Running", "Pause to inspect; reset to change the configuration.")
        self.viewer.timer.start()

    def pause_run(self):
        if self.state == "Running":
            self.viewer.timer.stop()
            self.viewer.refresh_last_frame()
            self._set_state("Paused", "Resume continues this run; reset starts over.")

    def _tick(self):
        if self.state != "Running":
            return
        try:
            with params_overridden(self.run_config.params_overrides):
                self.viewer._on_timer()
        except Exception as exc:
            self.viewer.timer.stop()
            self.viewer.refresh_last_frame()
            self._set_state("Error", "Reset before running again.")
            self._capture_result("Error", str(exc))
            self._show_error("Simulation stopped", exc)
            return
        if self.sim.done:
            self.viewer.timer.stop()
            self.viewer._run_complete = True
            self.viewer.refresh_last_frame()
            self._set_state("Done", "Run again, change configuration, or export the recorded result.")
            self._capture_result("Done")

    def _capture_result(self, state, error=""):
        if self.sim.k == 0:
            return
        self.last_result = RunResult.capture(self.run_config, self.sim, state, error)
        self.last_result_button.setEnabled(True)
        self.viewer.result_text.setText(self.last_result.summary())
        self.viewer.result_card.show()

    def show_last_result(self):
        if self.last_result is None:
            return
        if self.result_dialog is not None:
            self.result_dialog.close()
        dialog = QtWidgets.QDialog(self)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("Last result")
        dialog.resize(640, 300)
        layout = QtWidgets.QVBoxLayout(dialog)
        result = self.last_result
        title = QtWidgets.QLabel(f"{result.config['name']} · {registry.get(result.config['controller_mode']).label}")
        title.setWordWrap(True)
        layout.addWidget(title)
        text = QtWidgets.QLabel(result.summary())
        text.setWordWrap(True)
        text.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(text)
        export = QtWidgets.QPushButton("Export results…")
        export.clicked.connect(lambda: self.export_results(result=result))
        layout.addWidget(export)
        self.result_dialog = dialog
        dialog.destroyed.connect(lambda: setattr(self, "result_dialog", None)
                                 if self.result_dialog is dialog else None)
        dialog.show()

    def export_results(self, checked=False, *, result=None):
        result = result or self.last_result
        if result is None:
            return
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a parent directory for the run archive", str(archive.RUNS_ROOT))
        if not root:
            return
        try:
            saved = result.export(root)
        except Exception as exc:
            self._show_error("Could not export results", exc)
            return
        self.status.setText(f"Exported result to {saved}")

    def save_to(self, path):
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".yaml")
        if path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("Save the configuration with a .yaml or .yml extension.")
        raw = self.selected_config().to_dict()
        if raw["sim_mode"] == "trajectory" and raw["waypoint_source"] == "csv":
            # The simulator resolves CSV paths from the working directory.
            raw["waypoints_csv_path"] = str(Path(raw["waypoints_csv_path"]).resolve())
        data = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True).encode("utf-8")
        output = QtCore.QSaveFile(str(path))
        if not output.open(QtCore.QIODevice.OpenModeFlag.WriteOnly):
            raise OSError(output.errorString())
        if output.write(data) != len(data) or not output.commit():
            raise OSError(output.errorString())
        return path

    def save_config(self):
        dialog = QtWidgets.QFileDialog(self, "Save run config", str(Path.cwd()))
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter("YAML configs (*.yaml *.yml)")
        dialog.setDefaultSuffix("yaml")
        dialog.selectFile(self.config.name + "_custom.yaml")
        if dialog.exec():
            try:
                saved = self.save_to(dialog.selectedFiles()[0])
            except Exception as exc:
                self._show_error("Could not save config", exc)
                return
            self.status.setText("Saved {}".format(saved))

    def _show_error(self, title, exc):
        QtWidgets.QMessageBox.critical(self, title, str(exc))

    def closeEvent(self, event):
        self._discard_view()
        super().closeEvent(event)


def launch(config):
    app = QtWidgets.QApplication.instance()
    created_app = app is None
    if created_app:
        app = QtWidgets.QApplication(sys.argv)
    window = SimulationLauncher(config)
    window.show()
    if created_app:
        return app.exec()
    return window
