"""
Opti-Route Main Window
Phase 3 Subsystem: Desktop UI Integration (SRS F2, F9, F10, F11 & Use Case U3)

Constructs the primary PyQt6 desktop application window:
- Top Toolbar with Theme Switcher (#1E1E1E / #FAFAFA), Auto-Fit, and Zoom controls.
- Central interactive MapCanvas for graph visualization and route overlays.
- Left Control Sidebar for algorithm selection, parameters, and execution.
- Bottom Playback Bar for vehicle traversal animation controls.
- Right Analytics Dock placeholder reserved for Meghana's Matplotlib convergence engine (Use Case U4).
"""

from __future__ import annotations
from typing import Optional

try:
    from PyQt6.QtCore import Qt, pyqtSlot
    from PyQt6.QtGui import QAction, QFont, QIcon
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSlider,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QTabWidget,
        QTextEdit,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    def pyqtSlot(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator

from .map_canvas import MapCanvas
from .theme_manager import ThemeManager, ThemeMode


if HAS_PYQT:
    class OptiRouteMainWindow(QMainWindow):
        """
        Main Application Window for Opti-Route.
        Coordinates MVC Controller, Map Canvas, Theme System, and Subsystem Panels.
        """

        def __init__(
            self,
            controller=None,
            theme_manager: Optional[ThemeManager] = None,
            parent: Optional[QWidget] = None,
        ):
            if HAS_PYQT:
                super().__init__(parent)

            self.controller = controller
            self.theme_manager = theme_manager if theme_manager is not None else ThemeManager()

            if HAS_PYQT:
                self.setWindowTitle("Opti-Route: Adaptive Vehicle Routing & Task Dispatch (Group-15)")
                self.resize(1280, 800)
                self.setMinimumSize(960, 600)

                # Build UI Components
                self._init_canvas()
                self._init_toolbar()
                self._init_sidebar()
                self._init_bottom_playback_bar()
                self._init_analytics_panel()
                self._init_central_layout()
                self._init_statusbar()

                # Apply initial theme stylesheet and canvas colors
                self._apply_active_theme()

                # Populate graph into canvas if controller already has a graph
                if self.controller and getattr(self.controller, "graph", None):
                    self.load_graph(self.controller.graph)

        # -------------------------------------------------------------------------
        # Initialization Helpers
        # -------------------------------------------------------------------------

        def _init_canvas(self) -> None:
            """Initialize the interactive MapCanvas."""
            self.map_canvas = MapCanvas(theme_manager=self.theme_manager, parent=self)

        def _init_toolbar(self) -> None:
            """Construct top application toolbar."""
            toolbar = QToolBar("Main Controls", self)
            toolbar.setMovable(False)
            self.addToolBar(toolbar)

            # 1. Theme Toggle Button (SRS F11)
            self.theme_btn = QPushButton(self._theme_button_label(), self)
            self.theme_btn.setObjectName("toolbar_button")
            self.theme_btn.clicked.connect(self.on_toggle_theme)
            toolbar.addWidget(self.theme_btn)

            toolbar.addSeparator()

            # 2. Fit to View (SRS U3)
            self.fit_btn = QPushButton("Fit Map to View", self)
            self.fit_btn.setObjectName("toolbar_button")
            self.fit_btn.clicked.connect(self.map_canvas.fit_to_view)
            toolbar.addWidget(self.fit_btn)

            # 3. Zoom Controls
            self.zoom_in_btn = QPushButton("Zoom In (+)", self)
            self.zoom_in_btn.setObjectName("toolbar_button")
            self.zoom_in_btn.clicked.connect(self.map_canvas.zoom_in)
            toolbar.addWidget(self.zoom_in_btn)

            self.zoom_out_btn = QPushButton("Zoom Out (-)", self)
            self.zoom_out_btn.setObjectName("toolbar_button")
            self.zoom_out_btn.clicked.connect(self.map_canvas.zoom_out)
            toolbar.addWidget(self.zoom_out_btn)

            self.zoom_reset_btn = QPushButton("Reset Zoom", self)
            self.zoom_reset_btn.setObjectName("toolbar_button")
            self.zoom_reset_btn.clicked.connect(self.map_canvas.reset_zoom)
            toolbar.addWidget(self.zoom_reset_btn)

        def _init_sidebar(self) -> None:
            """Construct left sidebar control panel."""
            self.sidebar_frame = QFrame(self)
            self.sidebar_frame.setObjectName("control_panel")
            self.sidebar_frame.setFixedWidth(300)

            layout = QVBoxLayout(self.sidebar_frame)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(12)

            # Header
            title = QLabel("Routing Controls", self.sidebar_frame)
            title.setObjectName("heading_label")
            layout.addWidget(title)

            subtitle = QLabel("Phase 2 Solver Strategy Selection", self.sidebar_frame)
            subtitle.setObjectName("muted_label")
            layout.addWidget(subtitle)

            # 1. Algorithm Selection Group
            alg_group = QGroupBox("Optimization Strategy", self.sidebar_frame)
            alg_layout = QVBoxLayout(alg_group)

            self.alg_combo = QComboBox(alg_group)
            self.alg_combo.addItems([
                "Dijkstra Shortest Path",
                "Simulated Annealing (VRP/TSP)",
            ])
            self.alg_combo.currentIndexChanged.connect(self._on_algorithm_changed)
            alg_layout.addWidget(self.alg_combo)
            layout.addWidget(alg_group)

            # 2. Location Selection Group
            loc_group = QGroupBox("Locations", self.sidebar_frame)
            loc_layout = QGridLayout(loc_group)

            loc_layout.addWidget(QLabel("Start Node:"), 0, 0)
            self.start_node_combo = QComboBox(loc_group)
            loc_layout.addWidget(self.start_node_combo, 0, 1)

            loc_layout.addWidget(QLabel("Target Node:"), 1, 0)
            self.target_node_combo = QComboBox(loc_group)
            loc_layout.addWidget(self.target_node_combo, 1, 1)

            layout.addWidget(loc_group)

            # 3. Parameters Group (Simulated Annealing Hyper-parameters)
            self.params_group = QGroupBox("SA Hyper-parameters", self.sidebar_frame)
            params_layout = QGridLayout(self.params_group)

            params_layout.addWidget(QLabel("Initial Temp:"), 0, 0)
            self.init_temp_spin = QDoubleSpinBox(self.params_group)
            self.init_temp_spin.setRange(1.0, 10000.0)
            self.init_temp_spin.setValue(1000.0)
            params_layout.addWidget(self.init_temp_spin, 0, 1)

            params_layout.addWidget(QLabel("Cooling Rate:"), 1, 0)
            self.cooling_spin = QDoubleSpinBox(self.params_group)
            self.cooling_spin.setRange(0.01, 0.99)
            self.cooling_spin.setSingleStep(0.01)
            self.cooling_spin.setValue(0.95)
            params_layout.addWidget(self.cooling_spin, 1, 1)

            params_layout.addWidget(QLabel("Iterations:"), 2, 0)
            self.iter_spin = QSpinBox(self.params_group)
            self.iter_spin.setRange(1, 1000)
            self.iter_spin.setValue(50)
            params_layout.addWidget(self.iter_spin, 2, 1)

            layout.addWidget(self.params_group)
            self.params_group.setVisible(False)  # Hidden when Dijkstra is selected

            # 4. Execution Action Button
            self.run_btn = QPushButton("Calculate Route", self.sidebar_frame)
            self.run_btn.setObjectName("primary_button")
            self.run_btn.clicked.connect(self.on_run_solver)
            layout.addWidget(self.run_btn)

            # 5. Network Summary Card
            summary_card = QFrame(self.sidebar_frame)
            summary_card.setObjectName("card")
            summary_layout = QVBoxLayout(summary_card)
            summary_layout.setContentsMargins(8, 8, 8, 8)

            self.summary_title = QLabel("Network Topology", summary_card)
            self.summary_title.setObjectName("heading_label")
            summary_layout.addWidget(self.summary_title)

            self.node_count_lbl = QLabel("Nodes: 0", summary_card)
            summary_layout.addWidget(self.node_count_lbl)

            self.edge_count_lbl = QLabel("Road Segments: 0", summary_card)
            summary_layout.addWidget(self.edge_count_lbl)

            layout.addWidget(summary_card)
            layout.addStretch()

        def _init_bottom_playback_bar(self) -> None:
            """Construct bottom animation playback bar (SRS F10)."""
            self.playback_frame = QFrame(self)
            self.playback_frame.setObjectName("card")
            self.playback_frame.setFixedHeight(64)

            layout = QHBoxLayout(self.playback_frame)
            layout.setContentsMargins(14, 8, 14, 8)
            layout.setSpacing(12)

            # Playback control buttons
            self.play_btn = QPushButton("▶ Play", self.playback_frame)
            self.play_btn.setFixedWidth(80)
            layout.addWidget(self.play_btn)

            self.pause_btn = QPushButton("⏸ Pause", self.playback_frame)
            self.pause_btn.setFixedWidth(80)
            layout.addWidget(self.pause_btn)

            self.step_btn = QPushButton("⏭ Step", self.playback_frame)
            self.step_btn.setFixedWidth(80)
            layout.addWidget(self.step_btn)

            self.reset_btn = QPushButton("↺ Reset", self.playback_frame)
            self.reset_btn.setFixedWidth(80)
            layout.addWidget(self.reset_btn)

            # Traversal Progress
            layout.addWidget(QLabel("Progress:", self.playback_frame))
            self.progress_bar = QProgressBar(self.playback_frame)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFixedWidth(180)
            layout.addWidget(self.progress_bar)

            # Speed Multiplier Slider
            layout.addWidget(QLabel("Speed:", self.playback_frame))
            self.speed_slider = QSlider(Qt.Orientation.Horizontal, self.playback_frame)
            self.speed_slider.setRange(5, 50)  # 0.5x to 5.0x
            self.speed_slider.setValue(10)      # 1.0x default
            self.speed_slider.setFixedWidth(100)
            self.speed_slider.valueChanged.connect(self._on_speed_changed)
            layout.addWidget(self.speed_slider)

            self.speed_lbl = QLabel("1.0x", self.playback_frame)
            self.speed_lbl.setFixedWidth(36)
            layout.addWidget(self.speed_lbl)

            layout.addStretch()

        def _init_analytics_panel(self) -> None:
            """Construct integration container reserved for Meghana's convergence panel (U4)."""
            self.analytics_frame = QFrame(self)
            self.analytics_frame.setObjectName("control_panel")
            self.analytics_frame.setFixedWidth(280)

            layout = QVBoxLayout(self.analytics_frame)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            heading = QLabel("Live Analytics", self.analytics_frame)
            heading.setObjectName("heading_label")
            layout.addWidget(heading)

            subheading = QLabel("Use Case U4 Integration Slot (Meghana)", self.analytics_frame)
            subheading.setObjectName("muted_label")
            layout.addWidget(subheading)

            placeholder_card = QFrame(self.analytics_frame)
            placeholder_card.setObjectName("card")
            card_layout = QVBoxLayout(placeholder_card)

            info_text = (
                "<b>Performance & Convergence Engine</b><br><br>"
                "This slot is reserved for Meghana's Matplotlib convergence graphs and real-time "
                "algorithmic telemetry.<br><br>"
                "<i>Integration Hook:</i><br>"
                "<code>controllers.get_benchmark_payload()</code>"
            )
            info_lbl = QLabel(info_text, placeholder_card)
            info_lbl.setWordWrap(True)
            card_layout.addWidget(info_lbl)

            layout.addWidget(placeholder_card)
            layout.addStretch()

        def _init_central_layout(self) -> None:
            """Assemble main central layout with splitters."""
            central_widget = QWidget(self)
            self.setCentralWidget(central_widget)

            main_vbox = QVBoxLayout(central_widget)
            main_vbox.setContentsMargins(0, 0, 0, 0)
            main_vbox.setSpacing(0)

            # Horizontal Splitter: Sidebar | MapCanvas | Analytics
            h_splitter = QSplitter(Qt.Orientation.Horizontal, central_widget)
            h_splitter.addWidget(self.sidebar_frame)
            h_splitter.addWidget(self.map_canvas)
            h_splitter.addWidget(self.analytics_frame)
            h_splitter.setStretchFactor(0, 0)
            h_splitter.setStretchFactor(1, 1)
            h_splitter.setStretchFactor(2, 0)

            main_vbox.addWidget(h_splitter, stretch=1)
            main_vbox.addWidget(self.playback_frame, stretch=0)

        def _init_statusbar(self) -> None:
            """Construct status bar."""
            self.status_bar = QStatusBar(self)
            self.setStatusBar(self.status_bar)
            self._update_status_bar()

        # -------------------------------------------------------------------------
        # Event Handlers & State Updates
        # -------------------------------------------------------------------------

        def _theme_button_label(self) -> str:
            """Format theme button label based on current mode."""
            return "☀️ Switch to Light Theme" if self.theme_manager.is_dark else "🌙 Switch to Dark Theme"

        def _apply_active_theme(self) -> None:
            """Apply theme QSS to entire window and restyle canvas."""
            if not HAS_PYQT:
                return
            qApp = QApplication.instance()
            if isinstance(qApp, QApplication):
                qApp.setStyleSheet(self.theme_manager.get_stylesheet())
            self.map_canvas.apply_theme(self.theme_manager)
            self.theme_btn.setText(self._theme_button_label())
            self._update_status_bar()

        @pyqtSlot()
        def on_toggle_theme(self) -> None:
            """One-click theme toggle between Dark (#1E1E1E) and Light (#FAFAFA) (SRS F11)."""
            self.theme_manager.toggle_theme()
            self._apply_active_theme()

        def _on_algorithm_changed(self, index: int) -> None:
            """Show or hide SA hyper-parameter controls based on algorithm."""
            is_sa = (index == 1)
            self.params_group.setVisible(is_sa)

        def _on_speed_changed(self, value: int) -> None:
            """Update speed multiplier label."""
            speed_factor = value / 10.0
            self.speed_lbl.setText(f"{speed_factor:.1f}x")

        def _update_status_bar(self) -> None:
            """Update status bar information."""
            if not HAS_PYQT:
                return
            theme_str = "Dark Theme (#1E1E1E)" if self.theme_manager.is_dark else "Light Theme (#FAFAFA)"
            node_c = len(self.map_canvas._node_items)
            edge_c = len(self.map_canvas._edge_items)
            self.status_bar.showMessage(f"Opti-Route Ready | Theme: {theme_str} | Network: {node_c} nodes, {edge_c} edges")

        def load_graph(self, graph) -> None:
            """Load and display a RoadGraph model on the canvas and in UI selectors."""
            if graph is None:
                return

            self.map_canvas.set_graph(graph)

            if HAS_PYQT:
                # Update location selectors
                self.start_node_combo.clear()
                self.target_node_combo.clear()

                node_ids = sorted(list(graph.nodes.keys()))
                self.start_node_combo.addItems(node_ids)
                self.target_node_combo.addItems(node_ids)

                if len(node_ids) > 1:
                    self.target_node_combo.setCurrentIndex(len(node_ids) - 1)

                # Update summary card
                self.node_count_lbl.setText(f"Nodes: {graph.node_count()}")
                self.edge_count_lbl.setText(f"Road Segments: {graph.edge_count()}")
                self._update_status_bar()

        @pyqtSlot()
        def on_run_solver(self) -> None:
            """Execute selected algorithm via controller and overlay route result."""
            if not self.controller or not getattr(self.controller, "graph", None):
                return

            start_n = self.start_node_combo.currentText()
            target_n = self.target_node_combo.currentText()

            if not start_n or not target_n:
                return

            is_sa = (self.alg_combo.currentIndex() == 1)

            try:
                if is_sa:
                    all_nodes = [nid for nid in self.controller.graph.nodes.keys() if nid not in (start_n, target_n)]
                    result = self.controller.run_simulated_annealing(
                        start_node=start_n,
                        end_node=target_n,
                        waypoints=all_nodes,
                        initial_temp=self.init_temp_spin.value(),
                        cooling_rate=self.cooling_spin.value(),
                        iterations_per_temp=self.iter_spin.value(),
                    )
                else:
                    result = self.controller.run_dijkstra(start_node=start_n, end_node=target_n)

                if result and result.found:
                    self.map_canvas.draw_route(result.path, result.trajectory)
                    self.status_bar.showMessage(
                        f"{result.algorithm_name} Complete | Cost: {result.total_cost:.2f} | Time: {result.execution_time_ms:.2f}ms"
                    )
                else:
                    self.map_canvas.clear_routes()
                    self.status_bar.showMessage("No feasible route found.")
            except Exception as e:
                self.status_bar.showMessage(f"Solver Error: {str(e)}")

else:
    class OptiRouteMainWindow:
        """Fallback implementation when PyQt6 is not available."""

        def __init__(
            self,
            controller=None,
            theme_manager: Optional[ThemeManager] = None,
            parent: Optional[object] = None,
            *args,
            **kwargs,
        ) -> None:
            raise RuntimeError(
                "PyQt6 is required to instantiate OptiRouteMainWindow. "
                "Please install PyQt6 using 'pip install PyQt6'."
            )
