"""Create reusable Matplotlib figures from completed convergence histories."""

from typing import Iterable, Optional, Sequence, Tuple, Union

from .metrics_collector import BenchmarkResult


HistoryPoint = Union[Tuple[int, float], Tuple[int, float, float]]


def create_convergence_plot(
    history: Union[BenchmarkResult, Iterable[HistoryPoint]],
    title: str = "Simulated Annealing Convergence",
):
    """Return a Matplotlib Figure for normalized or solver-native history data.

    Solver-native points use ``(iteration, temperature, best_cost)``. Plotting
    completed histories is supported; the current solver exposes no live updates
    to this separate module.
    """
    from matplotlib.figure import Figure

    if isinstance(history, BenchmarkResult):
        points = list(history.convergence_history)
    else:
        points = list(history)

    iterations = []
    costs = []
    for point in points:
        if len(point) == 2:
            iteration, cost = point
        elif len(point) == 3:
            iteration, _, cost = point
        else:
            continue
        iterations.append(iteration)
        costs.append(cost)

    figure = Figure(figsize=(8, 4.5), constrained_layout=True)
    axes = figure.subplots()
    axes.set_title(title)
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Best-so-far solution cost")
    axes.grid(True, alpha=0.3)
    if iterations:
        axes.plot(iterations, costs, color="tab:blue", linewidth=1.5)
    else:
        axes.text(
            0.5,
            0.5,
            "No convergence history available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes.transAxes,
        )
    return figure
