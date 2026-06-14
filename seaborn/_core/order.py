from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas import Series

from seaborn._core.rules import variable_type


@dataclass
class _OrderEntry:
    """Internal entry for storing order information for a single variable."""
    user_order: list | None = None
    resolved_order: list | None = None
    is_explicit: bool = False
    source: str | None = None


class OrderRegistry:
    """
    Central registry for managing variable order resolution across seaborn.

    This class provides a unified interface for resolving and accessing the
    order of categorical levels for all plot variables (hue, style, size, row,
    col, x, y). It ensures consistency between figure-level APIs, FacetGrid,
    VectorPlotter, legend generation, and categorical axis ticks.

    The registry maintains two layers of order information:

    - **Global resolved order**: The canonical order for each variable, used
      by semantic mappings, legends, and row/col faceting. Every variable
      that appears in the plot gets a global resolved order, whether or not
      the user explicitly specified one.

    - **Axis-scope order**: Per-facet overrides for categorical axis
      variables (x, y). When ``sharex=False`` or ``sharey=False``, each
      facet may resolve its own tick order from local data. These local
      orders are recorded in the registry so they remain discoverable and
      don't rely on hidden local derivations.

    Parameters
    ----------
    data : DataFrame
        The plot data containing all variables.
    var_names : dict, optional
        Mapping from seaborn variable names (hue, style, size, row, col, x, y)
        to column names in the data.

    Examples
    --------
    Create a registry and resolve orders for multiple variables::

        registry = OrderRegistry(data, var_names={"hue": "species", "x": "sex"})
        registry.register("hue", order=["setosa", "versicolor", "virginica"])
        registry.register("x")

        hue_order = registry.get("hue")
        x_order = registry.get("x")

    Notes
    -----
    Order resolution priority (highest to lowest):
    1. User-explicitly provided order via ``register()``
    2. pandas Categorical dtype categories
    3. Unique values from data, sorted if numeric
    """

    _VALID_VARS = {"hue", "style", "size", "row", "col", "x", "y"}

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        var_names: dict[str, str | None] | None = None,
    ) -> None:
        self._data = data
        self._var_names = var_names or {}
        self._orders: dict[str, _OrderEntry] = {}
        self._axis_scopes: dict[tuple[str, int], list] = {}

    def _get_data_vector(self, var: str) -> Series:
        """Get the data vector for a given seaborn variable name."""
        if self._data is None:
            return Series(dtype=object)

        col_name = self._var_names.get(var, var)
        if col_name is None or col_name not in self._data.columns:
            return Series(dtype=object)

        return self._data[col_name]

    def register(
        self,
        var: str,
        order: list | None = None,
        *,
        data: Series | None = None,
    ) -> "OrderRegistry":
        """
        Register a variable and optionally its user-specified order.

        Parameters
        ----------
        var : str
            The seaborn variable name (hue, style, size, row, col, x, y).
        order : list, optional
            User-specified order for the variable levels.
        data : Series, optional
            Data vector to use for order resolution. If not provided, the
            registry's data will be used.

        Returns
        -------
        self : OrderRegistry
            For method chaining.
        """
        if var not in self._VALID_VARS:
            raise ValueError(
                f"Invalid variable name: {var}. "
                f"Must be one of {sorted(self._VALID_VARS)}"
            )

        entry = _OrderEntry(user_order=order, is_explicit=order is not None)
        self._orders[var] = entry

        if data is not None:
            self._resolve(var, data)

        return self

    def _resolve(self, var: str, data: Series) -> list:
        """
        Resolve the order for a variable using the provided data.

        Parameters
        ----------
        var : str
            The seaborn variable name.
        data : Series
            Data vector to use for order resolution.

        Returns
        -------
        order : list
            Resolved order of variable levels.
        """
        entry = self._orders.get(var, _OrderEntry())

        if entry.resolved_order is not None:
            return entry.resolved_order

        if entry.user_order is not None:
            order = list(entry.user_order)
        elif data.dtype.name == "category":
            order = list(data.cat.categories)
        else:
            order = list(filter(pd.notnull, data.unique()))
            if variable_type(pd.Series(order)) == "numeric":
                order.sort()

        order = list(filter(pd.notnull, order))
        entry.resolved_order = order
        entry.source = "user" if entry.is_explicit else (
            "categorical" if data.dtype.name == "category" else "data"
        )
        self._orders[var] = entry

        return order

    def resolve(self, var: str, data: Series | None = None) -> list:
        """
        Resolve and return the order for a variable.

        Parameters
        ----------
        var : str
            The seaborn variable name.
        data : Series, optional
            Data vector to use for order resolution. If not provided, the
            registry's data will be used.

        Returns
        -------
        order : list
            Resolved order of variable levels.
        """
        if data is None:
            data = self._get_data_vector(var)
        return self._resolve(var, data)

    def get(self, var: str, resolve: bool = True) -> list | None:
        """
        Get the resolved order for a variable.

        Parameters
        ----------
        var : str
            The seaborn variable name.
        resolve : bool, default True
            If True and the order hasn't been resolved yet, resolve it using
            the registry's data.

        Returns
        -------
        order : list or None
            Resolved order, or None if the variable is not registered and
            resolve is False.
        """
        if var not in self._orders:
            if resolve:
                return self.resolve(var)
            return None

        entry = self._orders[var]
        if entry.resolved_order is None and resolve:
            return self.resolve(var)

        return entry.resolved_order

    def is_explicit(self, var: str) -> bool:
        """
        Check if the order for a variable was explicitly provided by the user.

        Parameters
        ----------
        var : str
            The seaborn variable name.

        Returns
        -------
        is_explicit : bool
            True if the order was explicitly provided, False otherwise.
        """
        entry = self._orders.get(var)
        return entry.is_explicit if entry is not None else False

    def update_data(self, data: pd.DataFrame) -> None:
        """
        Update the registry's data reference.

        Parameters
        ----------
        data : DataFrame
            New data DataFrame.
        """
        self._data = data

    def update_resolved(self, var: str, order: list) -> None:
        """
        Update the resolved order for a variable.

        This is useful when the order needs to be transformed after initial
        resolution (e.g., when categorical values are converted to strings
        for matplotlib).

        Parameters
        ----------
        var : str
            The seaborn variable name.
        order : list
            The new resolved order to store.
        """
        if var in self._orders:
            self._orders[var].resolved_order = list(order)

    def update_var_names(self, var_names: dict[str, str | None]) -> None:
        """
        Update the registry's variable name mapping.

        Parameters
        ----------
        var_names : dict
            New mapping from seaborn variable names to data column names.
        """
        self._var_names.update(var_names)

    def set_axis_scope(self, var: str, axis_id: int, order: list) -> None:
        """
        Record a per-facet (axis-scope) order for a categorical axis variable.

        This is used when ``sharex=False`` or ``sharey=False``: each facet
        may resolve its own tick order from local data.  The local order is
        stored here so it is discoverable and does not rely on hidden
        local derivations.

        Parameters
        ----------
        var : str
            The seaborn variable name (typically ``"x"`` or ``"y"``).
        axis_id : int
            Identifier for the axis object (``id(axis)``).
        order : list
            The locally resolved order for this axis.
        """
        self._axis_scopes[(var, axis_id)] = list(order)

    def get_axis_scope(self, var: str, axis_id: int) -> list | None:
        """
        Retrieve a previously recorded per-facet order.

        Parameters
        ----------
        var : str
            The seaborn variable name (typically ``"x"`` or ``"y"``).
        axis_id : int
            Identifier for the axis object (``id(axis)``).

        Returns
        -------
        order : list or None
            The locally resolved order, or ``None`` if not recorded.
        """
        return self._axis_scopes.get((var, axis_id))

    def has_axis_scope(self, var: str, axis_id: int) -> bool:
        """Check whether an axis-scope order has been recorded."""
        return (var, axis_id) in self._axis_scopes

    def __contains__(self, var: str) -> bool:
        """Check if a variable is registered."""
        return var in self._orders

    def __getitem__(self, var: str) -> list:
        """Get the resolved order for a variable using dict syntax."""
        if var not in self._orders:
            raise KeyError(f"Variable '{var}' is not registered")
        order = self.get(var)
        if order is None:
            raise KeyError(f"Variable '{var}' has no resolved order")
        return order

    def __repr__(self) -> str:
        entries = []
        for var, entry in sorted(self._orders.items()):
            status = "explicit" if entry.is_explicit else "implicit"
            if entry.resolved_order is not None:
                n = len(entry.resolved_order)
                entries.append(f"{var}: {status} ({n} levels)")
            else:
                entries.append(f"{var}: {status} (unresolved)")
        n_scopes = len(self._axis_scopes)
        if n_scopes:
            entries.append(f"axis_scopes: {n_scopes}")
        return f"OrderRegistry({', '.join(entries)})"


def categorical_order(vector, order=None):
    """
    Return a list of unique data values using seaborn's ordering rules.

    This is a convenience wrapper that creates a temporary OrderRegistry
    and resolves the order for a single variable.

    Parameters
    ----------
    vector : list, array, Categorical, or Series
        Vector of "categorical" values.
    order : list, optional
        Desired order of category levels to override the order determined
        from the data object.

    Returns
    -------
    order : list
        Ordered list of category levels not including null values.
    """
    if order is not None:
        return list(order)

    if hasattr(vector, "categories"):
        order = vector.categories
    else:
        try:
            order = vector.cat.categories
        except (TypeError, AttributeError):

            order = pd.Series(vector).unique()

            if variable_type(pd.Series(vector)) == "numeric":
                order = np.sort(order)

        order = filter(pd.notnull, order)
    return list(order)
