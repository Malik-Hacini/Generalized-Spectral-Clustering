"""Metric specifications for proxy-based AMI correlation analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    """Proxy metric metadata used in analysis and model selection.

    Parameters
    ----------
    name : str
        Metric key as stored in benchmark outputs.
    display_name : str
        Human-readable metric name for plots/tables.
    optimize : str
        Optimization direction, either ``"max"`` or ``"min"``.
    """

    name: str
    display_name: str
    optimize: str

    def aligned_value(self, raw_value: float) -> float:
        """Return orientation-aligned value (higher is better for all metrics)."""
        if self.optimize == "max":
            return float(raw_value)
        if self.optimize == "min":
            return float(-raw_value)
        raise ValueError(f"Invalid optimize value for metric '{self.name}': {self.optimize}")

    @property
    def objective_text(self) -> str:
        return "higher is better" if self.optimize == "max" else "lower is better"


METRIC_REGISTRY: dict[str, MetricSpec] = {
    "graph_ch": MetricSpec(
        name="graph_ch",
        display_name="Graph-CH",
        optimize="max",
    ),
    "modularity": MetricSpec(
        name="modularity",
        display_name="Directed Modularity",
        optimize="max",
    ),
    "map_equation": MetricSpec(
        name="map_equation",
        display_name="Map Equation",
        optimize="min",
    ),
}


def get_metric_spec(metric_name: str) -> MetricSpec:
    """Get a registered metric specification by name."""
    if metric_name not in METRIC_REGISTRY:
        valid = ", ".join(sorted(METRIC_REGISTRY.keys()))
        raise ValueError(f"Unknown metric '{metric_name}'. Valid choices: {valid}")
    return METRIC_REGISTRY[metric_name]


def resolve_metric_specs(metric_names: list[str]) -> list[MetricSpec]:
    """Resolve an ordered list of metric names to metric specs."""
    seen = set()
    specs: list[MetricSpec] = []
    for name in metric_names:
        if name in seen:
            continue
        specs.append(get_metric_spec(name))
        seen.add(name)
    return specs
