#!/usr/bin/env python3
"""
Opti-Route: Adaptive Vehicle Routing & Task Dispatch System
Phase 3 GUI Application Launcher (Group-15)

Entry point establishing:
1. In-memory package bridge mapping 'opti-route' folder to 'opti_route' package.
2. Demonstration RoadGraph network initialization using actual domain models.
3. OptiRouteController integration.
4. PyQt6 QApplication and OptiRouteMainWindow launch with dual-theme system.
"""

import os
import sys
import types

# -----------------------------------------------------------------------------
# 1. In-Memory Package Bridge
# Resolves the directory name mismatch without renaming the project folder
# or modifying any pre-existing teammate import statements.
# -----------------------------------------------------------------------------
if "opti_route" not in sys.modules:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    pkg = types.ModuleType("opti_route")
    pkg.__path__ = [base_dir]
    sys.modules["opti_route"] = pkg

# -----------------------------------------------------------------------------
# 2. Domain Model & Controller Imports
# -----------------------------------------------------------------------------
from opti_route.models.graph import Node, RoadGraph
from opti_route.controllers import OptiRouteController
from opti_route.views import ThemeManager, OptiRouteMainWindow

# -----------------------------------------------------------------------------
# 3. Optional PyQt6 Availability Check
# QApplication is imported here at module scope so type-checkers see it as
# a stable binding. The headless fallback path (has_qt == False) is unchanged.
# -----------------------------------------------------------------------------
try:
    from PyQt6.QtWidgets import QApplication
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


def build_demo_graph() -> RoadGraph:
    """
    Constructs a realistic sample urban distribution network for GUI demonstration
    using the real RoadGraph and Node models.
    """
    graph = RoadGraph()

    # Define key logistics points (Depot, Distribution Hubs, Customer Delivery Points)
    nodes = [
        Node(id="Depot", x=100.0, y=250.0, name="Central Logistics Hub"),
        Node(id="Hub-North", x=280.0, y=100.0, name="North Transit Facility"),
        Node(id="Hub-South", x=300.0, y=420.0, name="South Distribution Depot"),
        Node(id="Cust-A", x=480.0, y=80.0, name="Metro Retailer A"),
        Node(id="Cust-B", x=520.0, y=220.0, name="Commercial Center B"),
        Node(id="Cust-C", x=490.0, y=360.0, name="Industrial Park C"),
        Node(id="Cust-D", x=700.0, y=150.0, name="Regional Warehouse D"),
        Node(id="Cust-E", x=680.0, y=380.0, name="Suburban Outlet E"),
        Node(id="Terminal", x=850.0, y=260.0, name="Port Terminal"),
    ]

    for n in nodes:
        graph.add_node(n)

    # Connect nodes with weighted road segments (Euclidean distance + road factor)
    edges = [
        ("Depot", "Hub-North", 12.5),
        ("Depot", "Hub-South", 14.0),
        ("Hub-North", "Cust-A", 15.2),
        ("Hub-North", "Cust-B", 11.0),
        ("Hub-South", "Cust-B", 13.4),
        ("Hub-South", "Cust-C", 12.0),
        ("Cust-A", "Cust-D", 16.8),
        ("Cust-B", "Cust-D", 14.5),
        ("Cust-B", "Cust-C", 10.2),
        ("Cust-C", "Cust-E", 13.0),
        ("Cust-D", "Terminal", 15.0),
        ("Cust-E", "Terminal", 14.2),
        ("Cust-D", "Cust-E", 18.0),
    ]

    for src, dst, weight in edges:
        graph.add_edge(src, dst, weight, directed=False)

    return graph


def main():
    """Application main entry point."""
    print("=" * 70)
    print("Opti-Route: Adaptive Vehicle Routing & Task Dispatch System")
    print("Course: CS1207 — Object Oriented Programming Lab | Team: Group-15")
    print("Phase 3: GUI Canvas, Dual Themes, and Animated Traversal (Ethan Naik)")
    print("=" * 70)

    # 1. Initialize Graph & Controller
    graph = build_demo_graph()
    print(f"✓ Network initialized: {graph.node_count()} nodes, {graph.edge_count()} road connections.")

    controller = OptiRouteController(graph=graph)
    print("✓ OptiRouteController initialized successfully.")

    # 2. Check GUI Environment
    has_qt = HAS_PYQT

    if not has_qt:
        print("\n[DIAGNOSTIC NOTICE]")
        print("PyQt6 is not installed in the current Python environment.")
        print("To launch the graphical window, install PyQt6 using:")
        print("    pip install PyQt6")
        print("\nVerifying controller calculation headlessly:")
        res = controller.run_dijkstra("Depot", "Terminal")
        print(f"✓ Headless Dijkstra Test: Found={res.found}, Path={res.path}, Cost={res.total_cost:.2f}")
        return 0

    # 3. Launch PyQt6 Desktop Application
    app = QApplication(sys.argv)
    theme_manager = ThemeManager()
    app.setStyleSheet(theme_manager.get_stylesheet())

    window = OptiRouteMainWindow(controller=controller, theme_manager=theme_manager)
    window.show()

    print("✓ OptiRouteMainWindow running. Entering Qt event loop.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
