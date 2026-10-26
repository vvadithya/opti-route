# Phase 4: Benchmarking and reports

The `benchmarking` package is independent of the GUI. Its runner calls the
existing `DijkstraSolver.solve` and `SimulatedAnnealingSolver.solve` interfaces
directly; it does not change or depend on the orchestrator's memory-tracing
implementation.

## Modules

- `performance_monitor.py` times one operation with `time.perf_counter()` and
  measures peak traced Python allocations with `tracemalloc`. It does not
  measure total process memory. When tracing was already active, it leaves that
  session running and reports peak memory as unavailable rather than resetting
  another component's tracing state.
- `metrics_collector.py` defines the validated `BenchmarkResult` structure and
  stores distinguishable runs. Unsupported measurements remain `None`.
- `benchmark_runner.py` creates the supplied solver types, passes the caller's
  graph and solver arguments, then records measurements and available result
  fields. It runs synchronously; a GUI must dispatch it to a worker thread.
- `convergence_plot.py` returns a reusable Matplotlib `Figure`. It supports
  empty data and completed history; the current Simulated Annealing public API
  returns history after the solve and has no live-plot callback.
- `algorithm_comparison.py` creates one comparison row per run. Runtime and
  traced memory are reported independently. Cost and distance are compared
  only among runs explicitly assigned the same `comparison_group`; distance
  additionally needs matching units.
- `report_generator.py` exports a results CSV plus a separate convergence CSV
  when history exists, or a PDF with configuration, measurements, limitations,
  and convergence charts. Existing output files are not overwritten. PDF
  generation uses the project's planned Matplotlib dependency; no PDF package
  is added.

## Running a benchmark

Install Matplotlib (needed for plots and PDF; CSV and measurement code otherwise
use the standard library):

```powershell
python -m pip install matplotlib
```

From the package's importable parent directory, pass an actual `RoadGraph`
instance and solver arguments. For example:

```python
from opti_route.benchmarking import BenchmarkRunner, MetricsCollector
from opti_route.models.graph import Node, RoadGraph

graph = RoadGraph()
graph.add_node(Node("A", 0.0, 0.0))
graph.add_node(Node("B", 1.0, 0.0))
graph.add_edge("A", "B", 1.0)

collector = MetricsCollector()
runner = BenchmarkRunner(collector)
result = runner.run(
    "dijkstra",
    graph,
    run_id="dijkstra-001",
    solve_kwargs={"start_node": "A", "end_node": "B"},
    distance_unit="graph weight units",
)
```

For Simulated Annealing, use `"simulated_annealing"` and pass supported
constructor settings in `solver_options`; put `start_node`, `end_node`, and
optional `waypoints` in `solve_kwargs`. To record kilometres, set the graph
distance unit and provide a truthful conversion function as `distance_to_km`.
Assign a common `comparison_group` only when runs use the same intended workload
and objective; Dijkstra is point-to-point while Simulated Annealing searches a
waypoint route.

```python
from opti_route.benchmarking import export_csv, export_pdf

export_csv("benchmark.csv", collector.results)
export_pdf(
    "benchmark.pdf",
    collector.results,
    configuration={"dataset": "your dataset", "seed": 42},
)
```

The CSV exporter returns the main CSV path and an optional `_convergence.csv`
sidecar path. Missing metrics are empty fields, not fabricated zeros. PDF and
CSV exports fail explicitly if a destination already exists or cannot be
written.

## Integration points still needed

The supplied `RoadGraph` supports `has_node`, `get_neighbors`, and `get_node`.
Dijkstra needs a graph plus start/end node identifiers. Simulated Annealing
needs a graph plus start/end identifiers and optional waypoint node IDs; its
`fleet` and `job_queue` parameters do not currently implement fleet/job
optimization. The supplied graph does not define a distance unit, so the
benchmark runner labels distances as unspecified graph units unless the caller
provides a unit and conversion. No real benchmark data is included.

Ram's actual fleet, job queue, demand matrix, and serialization interfaces are
not present. To integrate those models, provide the graph-compatible object
expected by the solvers, start/end node identifiers (or the solver-supported
depot attributes), waypoint node IDs for Simulated Annealing, and a documented
distance unit/conversion. The current solver API does not expose multi-vehicle
routes or consume a job queue in its optimization objective.

Ethan's GUI source is also not present, so no GUI files were changed. A future
results panel can display `MetricsCollector.get_results()` and
`AlgorithmComparison.compare(...)`, embed the returned plot figure in a
Matplotlib Qt canvas, and invoke report exports from a worker thread. The
benchmarking package imports no PyQt6 widgets and does not assume GUI class or
signal names.

## Tests

Run the Phase 4 unit and integration tests from an environment where the
repository is importable as `opti_route`:

```powershell
python -m unittest discover -s tests -p "test_benchmarking.py"
```

The solver integration test uses only a small synthetic graph created in the
test; it is test data, not a project benchmark.
