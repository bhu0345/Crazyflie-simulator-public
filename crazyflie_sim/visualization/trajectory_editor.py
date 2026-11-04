"""Table editor for timed trajectories (waypoint_source "timed") in the GUI."""

import csv

import numpy as np

from crazyflie_sim.sim.config import from_dict
from crazyflie_sim.visualization.quad_plot_pg import THEME, _load_qt_modules

pg, gl, QtCore, QtWidgets = _load_qt_modules()

COLUMNS = ("t [s]", "x [m]", "y [m]", "z [m]", "ψ [deg]")
CSV_HEADER = ("t", "x", "y", "z", "yaw_deg")
# Take off, fly a square with heading changes, turn in place and descend.
TEMPLATE_ROWS = (
    (3.0, 0.0, 0.0, 1.0, 0.0),
    (6.0, 0.8, 0.0, 1.0, 0.0),
    (9.0, 0.8, 0.8, 1.0, 90.0),
    (12.0, 0.0, 0.8, 1.0, 180.0),
    (15.0, 0.0, 0.0, 1.0, 90.0),
    (17.0, 0.0, 0.0, 1.0, 0.0),
    (20.0, 0.0, 0.0, 0.3, 0.0),
)
# The LQR is linearized about hover at yaw 0 and loses track beyond this.
LQR_YAW_LIMIT_DEG = 45.0

_TABLE_QSS = """
    QTableWidget {{
        background-color: {surface_alt}; color: {text}; gridline-color: {border};
        border: 1px solid {border_strong}; border-radius: 8px; font-size: 13px;
        selection-background-color: {accent_strong}; selection-color: #ffffff;
    }}
    QTableWidget QLineEdit {{ background-color: {bg}; color: {text}; border: 1px solid {accent}; }}
    QHeaderView {{ background-color: {surface}; border: none; }}
    QHeaderView::section {{
        background-color: {surface}; color: {muted}; border: none;
        border-bottom: 1px solid {border_strong}; padding: 6px; font-weight: 600;
    }}
    QTableCornerButton::section {{ background-color: {surface}; border: none; }}
"""


def editor_rows(scenario):
    """Rows [t, x, y, z, yaw_deg] to start editing from: the scenario's own
    timed waypoints, a speed-timed trajectory's waypoints at the times it
    reaches them, or the template."""
    if scenario.sim_mode != "trajectory":
        return [tuple(row) for row in TEMPLATE_ROWS]
    if scenario.waypoint_source == "timed" and scenario.timed_waypoints:
        return [(t, x, y, z, float(np.degrees(yaw))) for t, x, y, z, yaw in scenario.timed_waypoints]
    try:
        waypoints = scenario.waypoints()
    except (OSError, ValueError):
        return [tuple(row) for row in TEMPLATE_ROWS]
    lengths = np.linalg.norm(np.diff(waypoints[:, :3], axis=0), axis=1)
    times = np.concatenate(([0.0], np.cumsum(lengths) / float(scenario.trajectory_speed)))
    rows = [(float(t), *map(float, w[:3]), float(np.degrees(w[3]))) for t, w in zip(times, waypoints)]
    if np.linalg.norm(waypoints[0, :3] - np.asarray(scenario.initial_pos, dtype=float)) <= 1e-6:
        return rows[1:]  # the start pose at t = 0 is implied
    return [(t + 2.0, *rest) for t, *rest in rows]  # 2 s to reach the first waypoint


def read_csv(path):
    """Rows [t, x, y, z, yaw_deg] from a CSV file; a header line is skipped."""
    rows = []
    with open(path, newline="") as handle:
        for number, line in enumerate(csv.reader(handle), start=1):
            if not line or not "".join(line).strip():
                continue
            try:
                values = [float(v) for v in line]
            except ValueError:
                if not rows and number == 1:
                    continue
                raise ValueError(f"line {number}: expected numbers, got {line}") from None
            if len(values) != 5:
                raise ValueError(f"line {number}: expected 5 values t, x, y, z, yaw_deg")
            rows.append(tuple(values))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows([f"{v:g}" for v in row] for row in rows)


class TrajectoryEditor(QtWidgets.QDialog):
    """Edit [t, x, y, z, yaw] waypoints; run_config() is the run they fly."""

    def __init__(self, parent, config):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Trajectory")
        self.resize(600, 540)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        intro = QtWidgets.QLabel(
            "Each row is a waypoint the reference reaches at time t. Between two waypoints every "
            "axis and the yaw move monotonically, without overshoot. The flight starts at "
            "({:g}, {:g}, {:g}) m at t = 0 and holds the last waypoint for 2 s.".format(
                *config.scenario.initial_pos))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setStyleSheet(_TABLE_QSS.format(**THEME))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAccessibleName("Waypoints")
        layout.addWidget(self.table, 1)

        actions = QtWidgets.QHBoxLayout()
        for text, slot in (("Add row", self.add_row), ("Remove rows", self.remove_rows),
                           ("Import CSV…", self.import_csv), ("Export CSV…", self.export_csv)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.addButton("Apply", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.apply_button.setObjectName("primaryButton")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.set_rows(editor_rows(config.scenario))
        self.table.itemChanged.connect(self.validate)

    # ---- table ------------------------------------------------------------
    def set_rows(self, rows):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row in rows:
            self._append(row)
        self.table.blockSignals(False)
        self.validate()

    def _append(self, row):
        index = self.table.rowCount()
        self.table.insertRow(index)
        for column, value in enumerate(row):
            # mm, ms and millidegrees: 1.5708 rad reads as 90, not 90.0002.
            item = QtWidgets.QTableWidgetItem(f"{round(float(value), 3):g}")
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(index, column, item)

    def add_row(self):
        """Append a copy of the last waypoint 2 s later (or of the start)."""
        try:
            last = self.rows()[-1]
        except (ValueError, IndexError):
            last = (0.0, *self.config.scenario.initial_pos, 0.0)
        self.table.blockSignals(True)
        self._append((last[0] + 2.0, *last[1:]))
        self.table.blockSignals(False)
        self.table.scrollToBottom()
        self.validate()

    def remove_rows(self):
        selected = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in selected or [self.table.rowCount() - 1]:
            if row >= 0:
                self.table.removeRow(row)
        self.validate()

    def rows(self):
        """Rows [t, x, y, z, yaw_deg]; ValueError names the first bad cell."""
        rows = []
        for r in range(self.table.rowCount()):
            values = []
            for c, name in enumerate(COLUMNS):
                item = self.table.item(r, c)
                try:
                    value = float(item.text()) if item is not None else float("nan")
                except ValueError:
                    value = float("nan")
                if not np.isfinite(value):
                    raise ValueError(f"Row {r + 1}: {name} is not a number.")
                values.append(value)
            if rows and values[0] <= rows[-1][0]:
                raise ValueError(f"Row {r + 1}: t must be later than row {r} ({rows[-1][0]:g} s).")
            rows.append(tuple(values))
        if not rows:
            raise ValueError("Add at least one waypoint.")
        return rows

    # ---- result -----------------------------------------------------------
    def run_config(self):
        """The edited config: this trajectory, flown for its own length."""
        raw = self.config.to_dict()
        if raw["sim_mode"] != "trajectory" or raw["waypoint_source"] != "timed":
            raw["name"] = "timed_trajectory"
        raw.update(sim_mode="trajectory", waypoint_source="timed", duration_s=None,
                   timed_waypoints=[[t, x, y, z, float(np.radians(yaw))] for t, x, y, z, yaw in self.rows()])
        return from_dict(raw)

    def validate(self):
        """Check the table; Apply is enabled only for a trajectory that flies."""
        try:
            cfg = self.run_config()
            cfg.scenario.reference()
        except ValueError as exc:
            self.apply_button.setEnabled(False)
            self.status.setStyleSheet(f"color: {THEME['danger']};")
            self.status.setText(str(exc))
            return False
        rows = self.rows()
        text = f"{len(rows)} waypoints · run length {cfg.scenario.resolved_duration():g} s"
        peak_yaw = max(abs(row[4]) for row in rows)
        if self.config.controller_mode == "full_state_feedback" and peak_yaw > LQR_YAW_LIMIT_DEG:
            self.status.setStyleSheet(f"color: {THEME['warn']};")
            text += (f". The Full-state LQR is linearized at ψ = 0 and loses track beyond about "
                     f"±{LQR_YAW_LIMIT_DEG:g}° (this trajectory reaches {peak_yaw:g}°).")
        else:
            self.status.setStyleSheet(f"color: {THEME['muted']};")
        self.status.setText(text)
        self.apply_button.setEnabled(True)
        return True

    # ---- CSV --------------------------------------------------------------
    def import_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import waypoints", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            rows = read_csv(path)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Could not import waypoints", str(exc))
            return
        self.set_rows(rows)

    def export_csv(self):
        try:
            rows = self.rows()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Could not export waypoints", str(exc))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export waypoints", "waypoints.csv",
                                                        "CSV files (*.csv)")
        if not path:
            return
        try:
            write_csv(path, rows)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Could not export waypoints", str(exc))
