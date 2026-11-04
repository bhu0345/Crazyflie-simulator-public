"""Interactive lifecycle and config round trips; Qt is optional in headless CI."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crazyflie_sim.runsim import make_frame_source, parse_args
from crazyflie_sim.sim.config import load_config, params_overridden
from crazyflie_sim.sim.simulation import Simulation


REPO = Path(__file__).resolve().parents[1]
HAS_QT = all(importlib.util.find_spec(name) is not None for name in ("PyQt6", "pyqtgraph", "OpenGL"))


class TestGuiArguments(unittest.TestCase):
    def test_gui_is_opt_in_and_keeps_config_overrides(self):
        self.assertFalse(parse_args([]).gui)
        args = parse_args(["default.yaml", "--gui", "--controller", "baseline"])
        self.assertTrue(args.gui)
        self.assertEqual(args.controller, "baseline")
        for option in (["--record", "flight.mp4"], ["--archive"]):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--gui", *option])


@unittest.skipUnless(HAS_QT, "optional PyQt6/PyQtGraph/OpenGL dependencies are not installed")
class TestGuiLauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from crazyflie_sim.visualization import launcher

        cls.gui = launcher
        cls.app = launcher.QtWidgets.QApplication.instance() or launcher.QtWidgets.QApplication([])

    def setUp(self):
        errors = patch.object(self.gui.SimulationLauncher, "_show_error")
        self.errors = errors.start()
        self.addCleanup(errors.stop)
        self.cfg = load_config(REPO / "crazyflie_sim/configs/hover_wind_dryden.yaml")
        self.cfg.scenario.duration_s = 0.12
        self.cfg.scenario.wind.start_time = 0.0
        self.cfg.params_overrides["battery_sag_enabled"] = True
        self.window = self.gui.SimulationLauncher(self.cfg)
        self.addCleanup(self.close_window)
        self.errors.assert_not_called()

    def close_window(self):
        self.window.close()
        self.app.sendPostedEvents(None, self.gui.QtCore.QEvent.Type.DeferredDelete)

    def finish(self):
        self.window.start_run()
        for _ in range(20):
            if self.window.state != "Running":
                break
            self.window._tick()
        self.assertEqual(self.window.state, "Done")
        return self.window.sim.results()

    def test_pause_reset_and_existing_live_loop_are_identical(self):
        import crazyflie_sim.model.params as params

        original_sag = params.battery_sag_enabled
        self.assertEqual(self.window.sim.t, 0.0)
        self.window.start_button.click()
        self.window._tick()
        self.window.pause_button.click()
        paused_t = self.window.sim.t
        self.assertGreater(paused_t, 0.0)
        self.assertEqual(self.window.state, "Paused")
        self.assertFalse(self.window.viewer.timer.isActive())
        self.assertFalse(self.window.controller_combo.isEnabled())
        self.window._tick()
        self.assertEqual(self.window.sim.t, paused_t)
        actual = self.finish()
        self.assertEqual(params.battery_sag_enabled, original_sag)

        cfg = self.window.selected_config()
        with params_overridden(cfg.params_overrides):
            expected_sim = Simulation.from_scenario(cfg.scenario, cfg.controller_mode, cfg.gains())
            frame = make_frame_source(expected_sim, cfg.scenario, cfg.controller_mode)
            while not expected_sim.done:
                frame(0)
            expected = expected_sim.results()
        for field in ("t", "pos", "att", "motor_pwm"):
            np.testing.assert_array_equal(actual[field], expected[field])

        old_timer = self.window.viewer.timer
        self.window.reset_button.click()
        self.assertFalse(old_timer.isActive())
        self.assertEqual(self.window.sim.t, 0.0)
        self.assertTrue(self.window.controller_combo.isEnabled())
        repeated = self.finish()
        np.testing.assert_array_equal(actual["pos"], repeated["pos"])

    def test_selection_and_saved_config_round_trip(self):
        self.window.controller_combo.setCurrentIndex(self.window.controller_combo.findData("full_state_feedback"))
        self.window.profile_combo.setCurrentIndex(self.window.profile_combo.findData("crazyflie21_standard"))
        self.window.profile_combo.activated.emit(self.window.profile_combo.currentIndex())
        self.assertEqual(self.window.sim.mode, "full_state_feedback")
        self.assertEqual(self.window.sim.quad.mass, 0.034)
        self.assertEqual(self.window.sim.controller.mass, 0.034)
        with tempfile.TemporaryDirectory() as tmp:
            saved = self.window.save_to(Path(tmp) / "saved")
            cfg = load_config(saved)
            self.assertEqual(cfg.to_dict(), self.window.selected_config().to_dict())
            self.window._apply_config(cfg)
            self.assertEqual(self.window.profile_combo.currentData(), "crazyflie21_standard")
        self.errors.assert_not_called()

    def test_scenario_defaults_and_overrides_do_not_leak(self):
        import crazyflie_sim.model.params as params

        previous = params.battery_sag_enabled
        self.window.start_run()
        self.window._tick()
        self.window.reset_run()
        path = REPO / "crazyflie_sim/configs/default.yaml"
        self.window.scenario_combo.setCurrentIndex(self.window.scenario_combo.findData(str(path)))
        self.window.scenario_combo.activated.emit(self.window.scenario_combo.currentIndex())
        cfg = load_config(path)
        self.assertEqual(self.window.selected_config().to_dict(), cfg.to_dict())
        self.assertEqual(params.battery_sag_enabled, previous)
        self.errors.assert_not_called()

    def test_tick_error_restores_globals_and_stops_timer(self):
        import crazyflie_sim.model.params as params

        previous = params.battery_sag_enabled
        self.window.start_run()
        with patch.object(self.window.sim, "step", side_effect=ValueError("bad plant")):
            self.window._tick()
        self.assertEqual(params.battery_sag_enabled, previous)
        self.assertEqual(self.window.state, "Error")
        self.assertFalse(self.window.viewer.timer.isActive())
        self.assertFalse(self.window.start_button.isEnabled())
        self.errors.assert_called_once()
        self.window.reset_run()
        self.assertEqual(self.window.state, "Ready")

    def test_invalid_config_keeps_current_selection(self):
        original = self.window.selected_config().to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            path.write_text("controller_mode: missing_controller\n")
            self.window.scenario_combo.addItem(path.name, str(path))
            self.window.scenario_combo.setCurrentIndex(self.window.scenario_combo.count() - 1)
            self.window._select_scenario()
        self.assertEqual(self.window.selected_config().to_dict(), original)
        self.assertEqual(self.window.scenario_combo.currentData(), self.cfg.source)
        self.errors.assert_called_once()

    def test_prepare_error_restores_globals_and_allows_recovery(self):
        import crazyflie_sim.model.params as params

        previous = params.battery_sag_enabled
        with patch.object(self.gui.Simulation, "from_scenario", side_effect=ValueError("bad config")):
            self.window.reset_run()
        self.assertEqual(params.battery_sag_enabled, previous)
        self.assertEqual(self.window.state, "Error")
        self.assertIsNone(self.window.viewer)
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertTrue(self.window.scenario_combo.isEnabled())
        self.window.reset_run()
        self.assertEqual(self.window.state, "Ready")

    def test_done_allows_selection_and_run_again_preserves_last_result(self):
        actual = self.finish()
        snapshot = self.window.last_result
        self.assertIsNotNone(snapshot)
        self.assertTrue(self.window.scenario_combo.isEnabled())
        self.assertEqual(self.window.start_button.text(), "Run again")
        self.assertTrue(self.window.pause_button.isHidden())
        self.assertEqual(self.window.viewer.run_state.text(), "DONE")
        self.assertAlmostEqual(float(self.window.viewer.time_value.text().split()[0]), self.window.sim.t)
        self.assertAlmostEqual(self.window.viewer.time_history[-1], self.window.sim.t)
        self.window.start_button.click()
        self.assertEqual(self.window.state, "Running")
        self.assertEqual(self.window.sim.t, 0.0)
        self.assertIs(self.window.last_result, snapshot)
        np.testing.assert_array_equal(snapshot.run["pos"], actual["pos"])
        self.window.pause_run()
        self.assertEqual(self.window.viewer.run_state.text(), "PAUSED")
        self.window.viewer.refresh_last_frame()
        self.assertEqual(self.window.viewer.run_state.text(), "PAUSED")

    def test_result_survives_config_change_and_cancelled_or_failed_export(self):
        import json
        from crazyflie_sim.sim.archive import load_timeseries

        actual = self.finish()
        snapshot = self.window.last_result
        original_config = snapshot.config.copy()
        self.window._apply_config(load_config(REPO / "crazyflie_sim/configs/default.yaml"))
        self.assertIs(self.window.last_result, snapshot)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(self.gui.QtWidgets.QFileDialog, "getExistingDirectory", return_value=""):
                self.window.export_results()
            self.assertEqual(list(Path(tmp).iterdir()), [])
            with patch.object(self.gui.QtWidgets.QFileDialog, "getExistingDirectory", return_value=tmp):
                self.window.export_results()
            folder = next(Path(tmp).iterdir())
            self.assertEqual(load_config(folder / "config.yaml").to_dict(), original_config)
            np.testing.assert_array_equal(load_timeseries(folder / "timeseries.npz")["pos"], actual["pos"])
            self.assertEqual(json.loads((folder / "env.json").read_text())["extra"]["state"], "Done")
            with patch.object(self.gui.QtWidgets.QFileDialog, "getExistingDirectory", return_value=tmp), \
                 patch.object(type(snapshot), "export", side_effect=OSError("disk full")):
                self.window.export_results()
            self.errors.assert_called_once()
            self.assertIs(self.window.last_result, snapshot)

    def test_error_keeps_partial_data_for_export(self):
        self.window.start_run()
        self.window._tick()
        before = self.window.sim.results()["pos"].copy()
        with patch.object(self.window.sim, "step", side_effect=ValueError("bad plant")):
            self.window._tick()
        result = self.window.last_result
        self.assertEqual(result.state, "Error")
        self.assertIn("bad plant", result.summary())
        np.testing.assert_array_equal(result.run["pos"], before)
        self.window.reset_run()
        self.assertIs(self.window.last_result, result)

    def test_plot_history_inspect_cursor_and_expansion_do_not_step_simulation(self):
        viewer = self.window.viewer
        for t in (0.02, 2.0, 8.0, 12.0, 15.0):
            viewer._update_telemetry(t, np.array([t, 0., 0.]), t, t / 100,
                                     np.full(4, 0.1), None, None)
        self.assertEqual(viewer.time_history[0], 0.02)
        self.assertEqual(len(viewer.time_history), 5)
        self.assertEqual(self.window.sim.t, 0.0)
        np.testing.assert_allclose(viewer.err_plot.viewRange()[0], [5, 15])
        viewer.plot_mode.setCurrentText("Full run")
        np.testing.assert_allclose(viewer.err_plot.viewRange()[0], [0, 15])
        viewer.plot_mode.setCurrentText("Inspect")
        viewer.err_plot.setXRange(1, 4, padding=0)
        viewer._manual_range([True, True])
        self.assertFalse(viewer.auto_y.isChecked())
        viewer.err_plot.setYRange(0, 0.9, padding=0)
        viewer._update_telemetry(20.0, np.zeros(3), 20., 0.2, np.zeros(4), None, None)
        np.testing.assert_allclose(viewer.err_plot.viewRange()[0], [1, 4])
        np.testing.assert_allclose(viewer.err_plot.viewRange()[1], [0, 0.9])
        viewer.cursor_time.setValue(2)
        self.assertIn("2.000 s", viewer._plots[0]["readout"].text())
        viewer.cursor_next.click()
        self.assertEqual(viewer.cursor_time.value(), 8.0)
        self.assertEqual(viewer._plots[0]["cursor"].value(), 8.0)
        viewer._expand_plot(viewer._plots[0])
        self.assertEqual(len(viewer._expanded), 1)
        expanded = viewer._expanded[0][2]
        np.testing.assert_allclose(expanded.viewRange(), viewer.err_plot.viewRange())
        expanded.setRange(xRange=(2, 6), yRange=(0, 0.4), padding=0)
        viewer._manual_range([True, True], expanded)
        np.testing.assert_allclose(viewer.err_plot.viewRange(), [[2, 6], [0, 0.4]])
        np.testing.assert_allclose(expanded.viewRange(), viewer.err_plot.viewRange())
        viewer.close_expanded()
        self.assertEqual(len(viewer._expanded), 0)
        self.assertEqual(self.window.sim.t, 0.0)

    def test_layout_and_event_preview_fit_small_desktop(self):
        self.window.resize(1024, 768)
        self.window.ensurePolished()
        self.window.main_layout.activate()
        self.assertLessEqual(self.window.minimumSizeHint().width(), 1024)
        self.assertEqual(self.window.width(), 1024)
        self.assertEqual(self.window.viewer.splitter.count(), 2)
        self.assertIn("Wind", self.window.summary.text())
        self.assertEqual(self.window.sim.t, 0.0)
        self.assertEqual(self.window.viewer.time_history, [])
        self.assertEqual(self.window.viewer.fault_title.text(), "Planned event")
        self.assertIn("At start", self.window.viewer.fault_detail.text())
        cfg = load_config(REPO / "crazyflie_sim/configs/hover_rotor_fault_default.yaml")
        cfg.scenario.rotor_fault.rotor = [1, 2]
        self.window._apply_config(cfg)
        self.assertIn("Rotor 1, 2", self.window.summary.text())
        self.assertIn("Rotor 1, 2", self.window.viewer.fault_detail.text())
        self.assertEqual(self.window.viewer.fault_title.text(), "Planned event")

    def test_default_hovering_hides_disturbance_panels(self):
        cfg = load_config(REPO / "crazyflie_sim/configs/default.yaml")
        cfg.scenario.duration_s = 0.12
        self.window._apply_config(cfg)
        viewer = self.window.viewer
        self.assertIn("Take off and hover. Climb to (0, 0, 1) m in 2 s, no disturbances",
                      self.window.summary.text())
        self.assertFalse(viewer.show_tracking)
        self.assertIsNotNone(viewer.reference_marker)  # the takeoff climb is drawn
        self.assertFalse(viewer.show_wind)
        self.assertFalse(viewer.show_rotor)
        self.assertIsNone(viewer.force_plot)
        self.assertIsNone(viewer.event_time_s)
        self.assertTrue(viewer.fault_card.isHidden())
        self.assertEqual(self.window.sim.t, 0.)
        self.finish()
        result = self.window.last_result
        self.assertEqual(result.interval["start_s"], 0.)
        self.assertTrue(result.interval["event_reached"])
        self.assertIn("rms_err_m", result.metrics)
        self.assertNotIn("Not available", result.summary())
        self.assertNotIn("Event not reached", result.summary())

    def test_auto_y_only_expands_until_fit(self):
        viewer = self.window.viewer
        for t, value in ((1., 10.), (2., 0.1), (15., 0.2)):
            viewer._update_telemetry(t, np.zeros(3), 0., value, None, None, None)
        self.assertGreater(viewer.err_plot.viewRange()[1][1], 10)
        viewer.fit_button.click()
        self.assertLess(viewer.err_plot.viewRange()[1][1], 1)

    def test_chart_reset_view_returns_to_live_auto_y(self):
        viewer = self.window.viewer
        for t, value in ((1., 10.), (2., 0.1), (15., 0.2)):
            viewer._update_telemetry(t, np.zeros(3), 0., value, None, None, None)
        self.assertGreater(viewer.err_plot.viewRange()[1][1], 10)  # Auto Y kept the 10 m peak
        viewer.err_plot.setRange(xRange=(0., 3.), yRange=(-1., 50.), padding=0)
        viewer._manual_range((True, True), viewer.err_plot)
        viewer.cursor_time.setValue(1.)
        self.assertEqual(viewer.plot_mode.currentText(), "Inspect")
        self.assertFalse(viewer.auto_y.isChecked())
        viewer.reset_charts_button.click()
        self.assertEqual(viewer.plot_mode.currentText(), "Live 10 s")
        self.assertTrue(viewer.auto_y.isChecked())
        self.assertEqual(viewer.err_plot.viewRange()[0], [5., 15.])
        self.assertLess(viewer.err_plot.viewRange()[1][1], 1)  # fitted to the last 10 s only
        self.assertEqual(viewer.cursor_time.value(), 15.)

    def test_slow_playback_changes_only_timer_and_survives_reset(self):
        viewer = self.window.viewer
        config = self.window.selected_config().to_dict()
        viewer.playback_speed.setCurrentIndex(viewer.playback_speed.findData(0.25))
        self.assertEqual(viewer.timer.interval(), 80)
        self.window.start_run()
        self.window._tick()
        self.window.pause_run()
        self.assertAlmostEqual(self.window.sim.t, 0.02)
        self.assertAlmostEqual(self.window.sim.dt, 0.002)
        self.assertEqual(self.window.selected_config().to_dict(), config)
        self.window.reset_run()
        self.assertEqual(self.window.viewer.playback_speed.currentData(), 0.25)
        self.assertEqual(self.window.viewer.timer.interval(), 80)

    def test_nose_arrow_and_mesh_follow_body_yaw_while_thrust_axis_stays_vertical(self):
        from crazyflie_sim.model.quadcopter import Quadcopter
        from crazyflie_sim.visualization.quad_plot_pg import _frame_rotation_and_origin

        viewer = self.window.viewer
        for yaw in (0., np.pi / 2, np.pi):
            quad = Quadcopter((0., 0., 1.), (0., 0., yaw))
            rot, origin = _frame_rotation_and_origin(quad.world_frame())
            viewer._update_body(rot, origin, quad.world_frame(), animate=False)
            direction = viewer.nose_arrow.pos[1] - viewer.nose_arrow.pos[0]
            np.testing.assert_allclose(direction / np.linalg.norm(direction),
                                       [np.cos(yaw), np.sin(yaw), 0.], atol=1e-6)
            thrust = viewer.thrust_axis_line.pos[1] - viewer.thrust_axis_line.pos[0]
            np.testing.assert_allclose(thrust / np.linalg.norm(thrust), [0., 0., 1.], atol=1e-6)
            hub = viewer.body_components[0]
            np.testing.assert_allclose(hub["item"].opts["meshdata"].vertexes(),
                                       hub["verts"] @ quad.rotation_body_to_world().T + quad.position(), atol=1e-7)

    def test_scene_references_are_optional_and_survive_reset_without_stepping(self):
        from crazyflie_sim.model.quadcopter import Quadcopter
        from crazyflie_sim.visualization.quad_plot_pg import _frame_rotation_and_origin

        viewer = self.window.viewer
        self.assertFalse(viewer.show_world_axes.isChecked())
        self.assertTrue(all(not axis.visible() for axis in viewer.world_axes))
        self.assertTrue(all(label.isHidden() for label in viewer.axis_labels))
        self.assertFalse(viewer.tether_line.visible())
        self.assertTrue(viewer.origin_marker.visible())
        self.assertTrue(viewer.nose_arrow.visible())
        viewer.show_world_axes.setChecked(True)
        viewer.show_height_projection.setChecked(True)
        self.assertTrue(all(axis.visible() for axis in viewer.world_axes))
        self.assertTrue(all(not label.isHidden() for label in viewer.axis_labels))
        self.assertTrue(viewer.tether_line.visible())

        quad = Quadcopter((0.2, -0.3, 1.), (0., 0., 0.))
        rot, origin = _frame_rotation_and_origin(quad.world_frame())
        viewer._update_body(rot, origin, quad.world_frame(), animate=False)
        segments = viewer.tether_line.pos.reshape(-1, 2, 3)
        np.testing.assert_allclose(segments[:, :, :2],
                                   np.broadcast_to([0.2, -0.3], segments[:, :, :2].shape))
        self.assertEqual(segments[0, 0, 2], 0.)
        self.assertTrue(np.all(segments[1:, 0, 2] > segments[:-1, 1, 2]))
        self.assertTrue(np.all(segments[:, 1, 2] > segments[:, 0, 2]))
        self.assertLessEqual(segments[-1, 1, 2], 1.)
        self.assertEqual(viewer.tether_line.mode, "lines")
        self.assertEqual(self.window.sim.t, 0.)

        self.window.reset_run()
        viewer = self.window.viewer
        self.assertTrue(viewer.show_world_axes.isChecked())
        self.assertTrue(viewer.show_height_projection.isChecked())
        viewer.show_world_axes.setChecked(False)
        viewer.show_height_projection.setChecked(False)
        self.assertTrue(all(not axis.visible() for axis in viewer.world_axes))
        self.assertTrue(all(label.isHidden() for label in viewer.axis_labels))
        self.assertFalse(viewer.tether_line.visible())
        self.assertEqual(self.window.sim.t, 0.)

    def test_view_menu_toggles_each_scene_mark_and_keeps_the_choices(self):
        self.assertIn("wind_arrow", self.window.viewer.scene_toggles)
        self.assertNotIn("waypoints", self.window.viewer.scene_toggles)
        self.window._apply_config(load_config(REPO / "crazyflie_sim/configs/trajectory_timed.yaml"))
        viewer = self.window.viewer
        self.assertEqual(list(viewer.scene_toggles), [
            "nose_arrow", "thrust_axis", "flown_path", "reference_path", "reference_point",
            "waypoints", "world_axes", "height_projection"])
        marks = {"thrust_axis": ([viewer.thrust_axis_line], "thrust axis"),
                 "flown_path": ([viewer.trail_line], "flown path"),
                 "reference_path": ([viewer.reference_line], "reference path"),
                 "reference_point": ([viewer.reference_marker], "reference now"),
                 "waypoints": (viewer.waypoint_items, "waypoints")}
        for key, (items, legend) in marks.items():
            with self.subTest(mark=key):
                self.assertTrue(all(item.visible() for item in items))
                self.assertIn(legend, viewer.scene_legend.text())
                viewer.scene_toggles[key].setChecked(False)
                self.assertFalse(any(item.visible() for item in items))
                self.assertNotIn(legend, viewer.scene_legend.text())
        self.assertFalse(viewer.scene_legend.isHidden())  # the nose arrow is still listed
        viewer.scene_toggles["nose_arrow"].setChecked(False)
        self.assertTrue(viewer.scene_legend.isHidden())

        # Reset keeps every choice; another scenario keeps those it shares.
        self.window.reset_run()
        viewer = self.window.viewer
        self.assertFalse(any(viewer.scene_toggles[key].isChecked() for key in marks))
        self.assertFalse(viewer.reference_line.visible())
        self.window._apply_config(load_config(REPO / "crazyflie_sim/configs/hover_wind_dryden.yaml"))
        viewer = self.window.viewer
        self.assertFalse(viewer.scene_toggles["flown_path"].isChecked())
        self.assertFalse(viewer.trail_line.visible())
        self.assertTrue(viewer.scene_toggles["wind_arrow"].isChecked())
        self.assertTrue(viewer.wind_arrow.visible())
        self.errors.assert_not_called()

    def test_view_menu_stays_open_while_toggling_several_marks(self):
        from pyqtgraph.Qt import QtTest

        Qt = self.gui.QtCore.Qt
        viewer = self.window.viewer
        menu = viewer.scene_options.menu()
        menu.popup(self.gui.QtCore.QPoint(0, 0))
        self.app.processEvents()
        self.assertTrue(menu.isVisible())
        for key in ("flown_path", "thrust_axis", "world_axes"):
            action = viewer.scene_toggles[key]
            before = action.isChecked()
            QtTest.QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
            self.assertEqual(action.isChecked(), not before)
            self.assertTrue(menu.isVisible())
        self.assertFalse(viewer.trail_line.visible())
        self.assertTrue(all(axis.visible() for axis in viewer.world_axes))
        menu.setActiveAction(viewer.scene_toggles["flown_path"])
        QtTest.QTest.keyClick(menu, Qt.Key.Key_Space)
        self.assertTrue(viewer.scene_toggles["flown_path"].isChecked())
        self.assertTrue(menu.isVisible())
        QtTest.QTest.keyClick(menu, Qt.Key.Key_Escape)
        self.assertFalse(menu.isVisible())

    def test_nose_arrow_toggle_survives_reset_and_reset_view_restores_the_camera(self):
        viewer = self.window.viewer
        self.assertTrue(viewer.show_nose_arrow.isChecked())
        self.assertTrue(viewer.nose_arrow.visible())
        self.assertIn("nose (body +X)", viewer.scene_legend.text())
        viewer.show_nose_arrow.setChecked(False)
        self.assertFalse(viewer.nose_arrow.visible())
        self.assertNotIn("nose (body +X)", viewer.scene_legend.text())
        self.window.reset_run()
        viewer = self.window.viewer
        self.assertFalse(viewer.show_nose_arrow.isChecked())
        self.assertFalse(viewer.nose_arrow.visible())
        viewer.show_nose_arrow.setChecked(True)
        self.assertTrue(viewer.nose_arrow.visible())

        def camera():
            opts = viewer.view.opts
            center = opts["center"]
            return (opts["distance"], opts["elevation"], opts["azimuth"], center.x(), center.y(), center.z())

        default = camera()
        viewer.view.setCameraPosition(distance=20., elevation=90., azimuth=10.)
        viewer.view.opts["center"] = viewer.pg.Vector(3., 4., 5.)
        self.assertNotEqual(camera(), default)
        viewer.reset_view_button.click()
        self.assertEqual(camera(), default)

        viewer.view.resize(800, 600)
        viewer._update_axis_labels()
        button, hint = viewer.reset_view_button.geometry(), viewer.scene_hint.geometry()
        self.assertLess(button.right(), hint.left())
        self.assertEqual(hint.right(), 800 - 14 - 1)
        self.assertLessEqual(abs(button.center().y() - hint.center().y()), 1)

    def test_world_indicator_tracks_camera_rotation_only_and_stays_in_corner(self):
        viewer = self.window.viewer
        viewer.view.setCameraPosition(elevation=90, azimuth=-90)
        viewer._update_axis_labels()
        np.testing.assert_allclose(viewer.world_indicator.axes, np.eye(3), atol=1e-6)
        viewer.view.setCameraPosition(elevation=90, azimuth=0)
        viewer._update_axis_labels()
        expected = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(viewer.world_indicator.axes, expected, atol=1e-6)
        viewer.view.opts["center"] = viewer.pg.Vector(3., 4., 5.)
        viewer.view.setCameraPosition(distance=20.)
        self.window.start_run()
        self.window._tick()
        self.window.pause_run()
        viewer._update_axis_labels()
        np.testing.assert_allclose(viewer.world_indicator.axes, expected, atol=1e-6)
        paused_t = self.window.sim.t
        for width, height in ((320, 400), (800, 600)):
            viewer.view.resize(width, height)
            viewer._update_axis_labels()
            self.assertEqual(viewer.world_indicator.x() + viewer.world_indicator.width(), width - 14)
            self.assertEqual(viewer.world_indicator.y(), 14)
        viewer.show_height_projection.setChecked(True)
        viewer.show_world_axes.setChecked(True)
        self.assertEqual(self.window.sim.t, paused_t)

    def test_trajectory_editor_validates_and_flies_a_custom_timed_run(self):
        from crazyflie_sim.visualization.trajectory_editor import TEMPLATE_ROWS, TrajectoryEditor

        dialog = TrajectoryEditor(self.window, self.window.selected_config())
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.rows(), list(TEMPLATE_ROWS))  # not a trajectory: the template
        self.assertTrue(dialog.apply_button.isEnabled())
        dialog.table.item(2, 0).setText("5")
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertIn("Row 3", dialog.status.text())
        dialog.table.item(2, 0).setText("x")
        self.assertIn("Row 3: t [s] is not a number", dialog.status.text())
        dialog.set_rows([(2.0, 0.0, 0.0, 1.0, 0.0), (5.0, 0.5, 0.0, 1.0, 30.0)])
        dialog.add_row()
        self.assertEqual(dialog.rows()[-1], (7.0, 0.5, 0.0, 1.0, 30.0))
        self.assertTrue(dialog.apply_button.isEnabled())

        cfg = dialog.run_config()
        self.assertEqual((cfg.scenario.sim_mode, cfg.scenario.waypoint_source), ("trajectory", "timed"))
        self.assertAlmostEqual(cfg.scenario.timed_waypoints[1][4], np.radians(30.0))
        self.assertIsNone(cfg.source)
        self.window._apply_config(cfg)
        self.assertEqual(self.window.scenario_combo.currentText(), "custom (unsaved)")
        self.assertIn("3 timed waypoints", self.window.summary.text())
        self.assertAlmostEqual(self.window.sim.duration_s, 9.0)
        viewer = self.window.viewer
        self.assertTrue(viewer.show_tracking)
        self.window.start_run()
        for _ in range(3):
            self.window._tick()
        self.window.pause_run()
        plots = {spec["key"]: spec for spec in viewer._plots}
        reference = self.window.sim.reference(self.window.sim.t)
        np.testing.assert_allclose(plots["tracking_pos"]["values"][-1, 3:], reference.pos)
        np.testing.assert_allclose(plots["tracking_pos"]["values"][-1, :3], self.window.sim.quad.position())
        self.assertEqual(plots["tracking_yaw"]["values"].shape[1], 2)

        # Switching away and back keeps the unsaved trajectory; saving keeps it too.
        hovering = str((REPO / "crazyflie_sim/configs/hovering.yaml").resolve())
        self.window.reset_run()
        self.window.scenario_combo.setCurrentIndex(self.window.scenario_combo.findData(hovering))
        self.window.scenario_combo.activated.emit(self.window.scenario_combo.currentIndex())
        self.assertEqual(self.window.config.name, "hovering")
        self.window.scenario_combo.setCurrentIndex(self.window.scenario_combo.findData(self.gui.CUSTOM_SCENARIO))
        self.window.scenario_combo.activated.emit(self.window.scenario_combo.currentIndex())
        self.assertEqual(self.window.config.scenario.timed_waypoints, cfg.scenario.timed_waypoints)
        with tempfile.TemporaryDirectory() as tmp:
            saved = load_config(self.window.save_to(Path(tmp) / "timed"))
        self.assertEqual(saved.scenario.timed_waypoints, cfg.scenario.timed_waypoints)
        self.errors.assert_not_called()

    def test_propellers_show_partial_and_complete_rotor_faults(self):
        from crazyflie_sim.visualization.quad_plot_pg import _gl_color

        for case, state, color in (("power_loss_50pct", "partial", "warn"), ("failure", "failed", "danger")):
            with self.subTest(case=case):
                cfg = load_config(REPO / "crazyflie_sim/configs/hover_rotor_fault_default.yaml")
                cfg.scenario.rotor_fault.case = case
                cfg.scenario.rotor_fault.start_time_s = 0.04
                cfg.scenario.duration_s = 0.2
                self.window._apply_config(cfg)
                viewer = self.window.viewer
                self.assertEqual(viewer.prop_states, ("nominal",) * 4)
                self.window.start_run()
                for _ in range(4):
                    self.window._tick()
                before = viewer.prop_angles.copy()
                self.window._tick()
                turned = viewer.prop_angles - before
                self.assertEqual(viewer.prop_states, (state, "nominal", "nominal", "nominal"))
                props = [c for c in viewer.body_components if c["rotor"] == 0 and not c["blur"]]
                disc = [c for c in viewer.body_components if c["rotor"] == 0 and c["blur"]]
                np.testing.assert_allclose(props[0]["item"].opts["color"], _gl_color(self.gui.THEME[color]))
                self.assertLess(turned[0], np.min(turned[1:]))
                if state == "failed":
                    self.assertEqual(turned[0], 0.0)
                    self.assertFalse(disc[0]["item"].visible())
                self.window.pause_run()


if __name__ == "__main__":
    unittest.main()
