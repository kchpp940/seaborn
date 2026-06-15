"""Internal diagnostics collection and export mechanism.

This module provides a unified way to collect diagnostic information during
plot construction across the objects API (Plot/Plotter) and the traditional
API (VectorPlotter / FacetGrid). The diagnostics are not exposed to regular
users by default but can be accessed in debug mode or via test helpers.

Diagnostic sections:
    - variables: Variable resolution (source data, column names, types)
    - grouping: Grouping keys and their resolved orders
    - order: Full order registry state
    - layout: Facet/subplot grid structure and ordering
    - legend: Legend contents, sources, and entry order
    - layers: Per-layer draw functions, stats, marks, and moves

"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame


_DIAGNOSTICS_ENV_VAR = "SEABORN_DIAGNOSTICS"


def diagnostics_enabled() -> bool:
    """Check whether diagnostics collection is globally enabled."""
    return os.environ.get(_DIAGNOSTICS_ENV_VAR, "0") == "1"


@dataclass
class _VariableDiagnostics:
    """Diagnostics for variable resolution."""

    assignments: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, var: str, *, source: str | None = None,
               name: str | None = None, var_type: str | None = None,
               **extra: Any) -> None:
        """Record diagnostic info for a single variable."""
        entry: dict[str, Any] = {"source": source, "name": name, "type": var_type}
        entry.update(extra)
        self.assignments[var] = entry


@dataclass
class _GroupingDiagnostics:
    """Diagnostics for grouping keys and their orders."""

    keys: list[str] = field(default_factory=list)
    orders: dict[str, list | None] = field(default_factory=dict)
    source: str | None = None

    def record_keys(self, keys: list[str]) -> None:
        """Record the list of grouping variable names."""
        self.keys = list(keys)

    def record_order(self, var: str, order: list | None) -> None:
        """Record the resolved order for a grouping variable."""
        self.orders[var] = list(order) if order is not None else None


@dataclass
class _OrderDiagnostics:
    """Diagnostics for the order registry."""

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, var: str, *, resolved_order: list | None = None,
               user_order: list | None = None, is_explicit: bool = False,
               source: str | None = None) -> None:
        """Record order info for a variable."""
        self.entries[var] = {
            "resolved_order": list(resolved_order) if resolved_order else None,
            "user_order": list(user_order) if user_order else None,
            "is_explicit": is_explicit,
            "source": source,
        }


@dataclass
class _LayoutDiagnostics:
    """Diagnostics for facet/subplot layout."""

    kind: str | None = None
    nrows: int = 0
    ncols: int = 0
    n_subplots: int = 0
    col_names: list[str] = field(default_factory=list)
    row_names: list[str] = field(default_factory=list)
    col_wrap: int | None = None
    sharex: bool | str = True
    sharey: bool | str = True
    subplots: list[dict[str, Any]] = field(default_factory=list)

    def record_subplot(self, subplot_info: dict[str, Any]) -> None:
        """Record info for a single subplot."""
        self.subplots.append(dict(subplot_info))


@dataclass
class _LegendDiagnostics:
    """Diagnostics for legend construction."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def record_entry(self, title: str, labels: list[str],
                     source: str | None = None) -> None:
        """Record a legend entry group."""
        self.entries.append({
            "title": title,
            "labels": list(labels),
            "source": source,
        })
        if source and source not in self.sources:
            self.sources.append(source)


@dataclass
class _LayerDiagnostics:
    """Diagnostics for a single plot layer."""

    kind: str | None = None
    mark: str | None = None
    stat: str | None = None
    moves: list[str] = field(default_factory=list)
    orient: str | None = None
    grouping_vars: list[str] = field(default_factory=list)
    legend: bool = True
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "mark": self.mark,
            "stat": self.stat,
            "moves": list(self.moves),
            "orient": self.orient,
            "grouping_vars": list(self.grouping_vars),
            "legend": self.legend,
            "label": self.label,
        }


class PlotDiagnostics:
    """Unified diagnostics collector for seaborn plot construction.

    This class collects diagnostic information across the full plot
    construction pipeline: variable resolution, grouping keys, order
    resolution, FacetGrid/subplot layout, legend construction, and the
    final draw functions/marks used.

    The collector is designed to be lightweight: when disabled, method
    calls are no-ops and minimal state is stored.

    Examples
    --------
    Enable via environment variable::

        import os
        os.environ["SEABORN_DIAGNOSTICS"] = "1"

    Access via test helper::

        from seaborn._testing import get_diagnostics
        g = sns.displot(data=tips, x="total_bill", hue="sex")
        diag = get_diagnostics(g)
        print(diag["variables"])
        print(diag["order"])
        print(diag["legend"])

    """

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = diagnostics_enabled()
        self._enabled = enabled

        self.variables = _VariableDiagnostics()
        self.grouping = _GroupingDiagnostics()
        self.order = _OrderDiagnostics()
        self.layout = _LayoutDiagnostics()
        self.legend = _LegendDiagnostics()
        self._layers: list[_LayerDiagnostics] = []

    # -- Enable / disable ------------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        """Whether diagnostics collection is currently enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable diagnostics collection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable diagnostics collection (existing data is preserved)."""
        self._enabled = False

    # -- Layer management ------------------------------------------------------ #

    def add_layer(self) -> _LayerDiagnostics:
        """Create and register a new layer diagnostics entry.

        Returns
        -------
        layer : _LayerDiagnostics
            The new layer diagnostics object for in-place updates.

        """
        layer = _LayerDiagnostics()
        if self._enabled:
            self._layers.append(layer)
        return layer

    @property
    def layers(self) -> list[_LayerDiagnostics]:
        """List of per-layer diagnostics entries."""
        return list(self._layers)

    # -- Export ---------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Export all diagnostics as a plain dictionary.

        Returns
        -------
        diagnostics : dict
            Dictionary with keys: "variables", "grouping", "order",
            "layout", "legend", "layers".

        """
        return {
            "variables": {
                "assignments": dict(self.variables.assignments),
            },
            "grouping": {
                "keys": list(self.grouping.keys),
                "orders": {k: list(v) if v else None for k, v in self.grouping.orders.items()},
                "source": self.grouping.source,
            },
            "order": {
                "entries": {
                    var: {
                        "resolved_order": list(info["resolved_order"]) if info["resolved_order"] else None,
                        "user_order": list(info["user_order"]) if info["user_order"] else None,
                        "is_explicit": info["is_explicit"],
                        "source": info["source"],
                    }
                    for var, info in self.order.entries.items()
                },
            },
            "layout": {
                "kind": self.layout.kind,
                "nrows": self.layout.nrows,
                "ncols": self.layout.ncols,
                "n_subplots": self.layout.n_subplots,
                "col_names": list(self.layout.col_names),
                "row_names": list(self.layout.row_names),
                "col_wrap": self.layout.col_wrap,
                "sharex": self.layout.sharex,
                "sharey": self.layout.sharey,
                "subplots": [dict(sp) for sp in self.layout.subplots],
            },
            "legend": {
                "entries": [
                    {
                        "title": e["title"],
                        "labels": list(e["labels"]),
                        "source": e["source"],
                    }
                    for e in self.legend.entries
                ],
                "sources": list(self.legend.sources),
            },
            "layers": [layer.to_dict() for layer in self._layers],
        }

    def __repr__(self) -> str:
        if not self._enabled:
            return "PlotDiagnostics(enabled=False)"
        return (
            f"PlotDiagnostics(enabled=True, "
            f"variables={len(self.variables.assignments)}, "
            f"layers={len(self._layers)}, "
            f"legend_entries={len(self.legend.entries)})"
        )
