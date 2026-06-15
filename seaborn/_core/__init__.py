"""
Internal core module for seaborn's declarative objects interface.

**Import boundary rules:**

- Upper-layer modules (``_base``, ``axisgrid``, ``categorical``, ``utils``,
  etc.) should import shared helpers **only** from this ``__init__`` module,
  not from individual ``_core`` sub-modules.
- Sub-modules whose names are **not** listed in ``__all__`` below are
  considered *private to* ``_core`` — they may be freely refactored
- The objects interface (``objects.py``) is exempt: it re-exports public
  API classes (``Plot``, ``Scale``, ``Move``, …) directly from their
  defining sub-modules for ``from seaborn.objects import …`` usage.
"""

from seaborn._core.rules import VarType, variable_type, categorical_order
from seaborn._core.typing import (
    Default,
    Deprecated,
    default,
    deprecated,
    ColumnName,
    DataSource,
    VariableSpec,
    VariableSpecList,
    OrderSpec,
    NormSpec,
    PaletteSpec,
    DiscreteValueSpec,
    ContinuousValueSpec,
    Vector,
)
from seaborn._core.data import PlotData, handle_data_source
from seaborn._core.exceptions import PlotSpecError
from seaborn._core.order import OrderRegistry

__all__ = [
    "VarType",
    "variable_type",
    "categorical_order",
    "Default",
    "Deprecated",
    "default",
    "deprecated",
    "ColumnName",
    "DataSource",
    "VariableSpec",
    "VariableSpecList",
    "OrderSpec",
    "NormSpec",
    "PaletteSpec",
    "DiscreteValueSpec",
    "ContinuousValueSpec",
    "Vector",
    "PlotData",
    "handle_data_source",
    "PlotSpecError",
    "OrderRegistry",
]
