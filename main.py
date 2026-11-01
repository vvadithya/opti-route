#!/usr/bin/env python3
"""
Opti-Route: Adaptive Vehicle Routing & Task Dispatch System
Full 4-Phase Integrated Application Launcher (Group-15)

Entry point establishing:
1. In-memory package bridge mapping 'opti-route' folder to 'opti_route' package.
2. Phase 1 Domain Model initialization (RoadGraph, VehicleFleet, DynamicJobQueue, GraphFactory).
3. Phase 2 & 4 OptiRouteController, Solvers (Dijkstra & Simulated Annealing), and Benchmarking integration.
4. Phase 3 PyQt6 QApplication and OptiRouteMainWindow launch with dual-theme system.
"""

import os
import sys
import types
from pathlib import Path

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
from opti_route.models.graph import Node, NodeType, RoadGraph
from opti_route.models.vehicle import Vehicle, VehicleFleet
from opti_route.models.job import DynamicJobQueue, Job
from opti_route.models.graph_factory import GraphFactory
from opti_route.controllers import OptiRouteController

# -----------------------------------------------------------------------------
# 3. Qt GUI Availability Check (Supports PyQt6 natively + PyQt5 bridge)
# -----------------------------------------------------------------------------
try:
    from opti_route.views import ensure_qt_bridge
    ensure_qt_bridge()
    from PyQt6.QtWidgets import QApplication
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


def build_demo_graph() -> RoadGraph:
    """
    Constructs a realistic sample urban distribution network for GUI demonstration
    using the real RoadGraph and Node models.
    """
    graph = RoadGraph(distance_unit="km")

    nodes = [
        Node(id="Depot", x=100.0, y=250.0, name="Central Logistics Hub", node_type=NodeType.DEPOT),
        Node(id="Hub-North", x=280.0, y=100.0, name="North Transit Facility", node_type=NodeType.HUB),
        Node(id="Hub-South", x=300.0, y=420.0, name="South Distribution Depot", node_type=NodeType.HUB),
        Node(id="Cust-A", x=480.0, y=80.0, name="Metro Retailer A", node_type=NodeType.JUNCTION),
        Node(id="Cust-B", x=520.0, y=220.0, name="Commercial Center B", node_type=NodeType.JUNCTION),
        Node(id="Cust-C", x=490.0, y=360.0, name="Industrial Park C", node_type=NodeType.JUNCTION),
        Node(id="Cust-D", x=700.0, y=150.0, name="Regional Warehouse D", node_type=NodeType.JUNCTION),
        Node(id="Cust-E", x=680.0, y=380.0, name="Suburban Outlet E", node_type=NodeType.JUNCTION),
        Node(id="Terminal", x=850.0, y=260.0, name="Port Terminal", node_type=NodeType.HUB),
    ]

    for n in nodes:
        graph.add_node(n)

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


def build_demo_fleet() -> VehicleFleet:
    """Constructs a sample delivery vehicle fleet."""
    fleet = VehicleFleet()
    fleet.add(Vehicle(id="V1", capacity=80.0, speed=60.0, fuel_rate=0.12, home_depot="Depot"))
    fleet.add(Vehicle(id="V2", capacity=100.0, speed=55.0, fuel_rate=0.15, home_depot="Depot"))
    return fleet


def build_demo_jobs() -> DynamicJobQueue:
    """Constructs a sample delivery job queue."""
    queue = DynamicJobQueue()
    queue.add(Job(id="J1", pickup="Hub-North", dropoff="Cust-A", weight=25.0, deadline=2.5))
    queue.add(Job(id="J2", pickup="Hub-North", dropoff="Cust-D", weight=35.0, deadline=3.5))
    queue.add(Job(id="J3", pickup="Hub-South", dropoff="Cust-C", weight=30.0, deadline=3.0))
    queue.add(Job(id="J4", pickup="Cust-B", dropoff="Cust-E", weight=20.0, deadline=4.5))
    queue.add(Job(id="J5", pickup="Cust-D", dropoff="Terminal", weight=40.0, deadline=5.0))
    return queue


def main():
    """Application main entry point."""
    print("=" * 70)
    print("Opti-Route: Adaptive Vehicle Routing & Task Dispatch System")
    print("Course: CS1207 — Object Oriented Programming Lab | Team: Group-15")
    print("Full 4-Phase Integrated Architecture (Models, Solvers, GUI, Benchmarks)")
    print("=" * 70)

    # 1. Initialize Phase 1 Models & Controller (via sample_network.json if present)
    sample_json = Path(__file__).resolve().parent / "data" / "sample_network.json"
    controller = OptiRouteController()

    if sample_json.exists():
        project = controller.load_project(sample_json)
        graph = project.graph
        print(
            f"✓ Loaded '{sample_json.name}' via GraphFactory: "
            f"{graph.node_count()} nodes, {graph.edge_count()} road segments, "
            f"{len(project.fleet)} vehicles, {len(project.jobs)} delivery jobs."
        )
    else:
        graph = build_demo_graph()
        controller.set_graph(graph)
        controller.set_fleet(build_demo_fleet())
        controller.set_job_queue(build_demo_jobs())
        print(f"✓ Network initialized: {graph.node_count()} nodes, {graph.edge_count()} road segments.")

    print("✓ OptiRouteController initialized with Phase 1–4 pipeline integration.")

    # 2. Check GUI Environment
    has_qt = HAS_PYQT

    if not has_qt:
        print("\n[DIAGNOSTIC NOTICE]")
        print("PyQt6 is not installed in the current Python environment.")
        print("To launch the graphical window, install dependencies using:")
        print("    pip install -r requirements.txt")
        print("\nVerifying full 4-phase controller pipeline headlessly:")
        res_dijkstra = controller.run_dijkstra("Depot", "Terminal")
        print(
            f"✓ Phase 2 Dijkstra: Found={res_dijkstra.found}, Path={res_dijkstra.path}, "
            f"Cost={res_dijkstra.total_cost:.2f}, Time={res_dijkstra.execution_time_ms:.2f}ms"
        )
        res_sa = controller.run_simulated_annealing("Depot", "Terminal", seed=42)
        print(
            f"✓ Phase 2 Simulated Annealing (Job Queue Waypoints): Found={res_sa.found}, "
            f"Cost={res_sa.total_cost:.2f}, Vehicles Routed={list(res_sa.routes.keys())}, "
            f"Time={res_sa.execution_time_ms:.2f}ms"
        )
        rows = controller.get_comparison_rows()
        print(f"✓ Phase 4 Benchmarking Collector: {len(rows)} runs recorded and compared.")
        return 0

    # Import GUI modules only after confirming PyQt6 is available.
    from opti_route.views import ThemeManager, OptiRouteMainWindow

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
