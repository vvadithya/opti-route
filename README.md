# opti-route

Vehicle routing and task dispatch project for CS1207 OOP Lab (Group 15).

You give it a road graph, a couple of vehicles, and a list of delivery jobs. It figures out the route order and draws the result on a Qt canvas so you can watch the vehicles move around or drag nodes to see the paths update.

## how it's structured

The repo is split into four phases:

Phase 1 (`models/`) handles the core data structures. `RoadGraph` stores nodes and weighted edges with an adjacency matrix cache. `VehicleFleet` and `DynamicJobQueue` hold the trucks and delivery tasks. `GraphFactory` reads and writes the whole state to JSON or CSV.

Phase 2 (`solvers/` and `controllers/`) is the routing logic. Every algorithm inherits from `BaseSolver` and gets called through `SolverOrchestrator`. Simulated Annealing, TLBO, and Genetic Algorithm do the actual VRP stop-reordering using 2-opt, swap, insertion, and order crossover. Dijkstra and Bellman-Ford are in there too—they handle point-to-point shortest paths under the hood and show up in the UI as unoptimized FIFO baselines so you have something to compare the heuristics against.

Phase 3 (`views/`) is the PyQt desktop app. Has a map canvas, a dark/light theme toggle, playback controls for the vehicle animation, and dialogs to add nodes, roads, vehicles, or jobs on the fly.

Phase 4 (`benchmarking/`) wraps solver runs with `time.perf_counter` and `tracemalloc` to log runtime, memory, and cost convergence, and dumps the comparison table to CSV or PDF if needed.

## running it

```bash
cd opti-route
python3 main.py
```

If you want to install the optional packages (`PyQt6`, `numpy`, `matplotlib`) first:

```bash
pip install -r requirements.txt
```

If you don't have `PyQt6`, `numpy`, or `matplotlib` installed, it still runs fine as long as `PyQt5` is on the system. `views/__init__.py` shims the Qt imports automatically and the rest falls back to standard library code.

## sample networks

There are 11 sample maps in `data/` that you can pick from the dropdown in the top bar.

The first few (`sample_network.json` and `sample_01` through `sample_08`) are smaller 8 to 15 node networks for quick testing. `sample_09_mega_metropolis_45nodes.json` and `sample_10_national_freight_60nodes.json` are larger grids with 45 and 60 nodes. On those two, running the FIFO Dijkstra/Bellman-Ford baseline gives a route cost around 1530 km and 1920 km because it just visits jobs in arrival order, while SA, TLBO, and GA bring it down to ~417 km and ~490 km.

## tests

```bash
python3 -m unittest discover tests -v
```
