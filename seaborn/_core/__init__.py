"""
Internal core module for seaborn's declarative objects interface.

**Import boundary rules:**

1. **Within ``_core`` itself** (``rules.py``, ``typing.py``, etc.):
   Use **relative imports** (``from .rules import X``). Never import from
   ``seaborn._core`` (the package root) to avoid circular dependencies.

2. **All modules outside ``_core``** — including ``objects.py``,
   ``_marks/*``, ``_stats/*``, ``_base.py``, ``axisgrid.py``, etc. —
   must import ``_core`` symbols **only** from this ``__init__`` module,
   not from individual ``_core`` sub-modules.

   Symbols not needed at ``import seaborn`` time are lazily loaded via
   ``__getattr__`` to break circular dependency chains (e.g.
   ``seaborn.palettes → seaborn.utils → seaborn._core → …``).

Sub-modules whose symbols are **not** listed in ``__all__`` below are
considered *private to* ``_core`` — they may be freely refactored.
"""

from __future__ import annotations

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

from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from seaborn._core.groupby import GroupBy
    from seaborn._core.scales import (
        Scale,
        Boolean,
        Continuous,
        Nominal,
        Temporal,
    )
    from seaborn._core.moves import Move, Dodge, Jitter, Norm, Shift, Stack
    from seaborn._core.plot import Plot
    from seaborn._core.properties import (
        PROPERTIES,
        Property,
        RGBATuple,
        DashPattern,
        DashPatternWithOffset,
    )


def __getattr__(name: str):
    _LAZY = {
        "GroupBy": "seaborn._core.groupby:GroupBy",
        "Scale": "seaborn._core.scales:Scale",
        "Boolean": "seaborn._core.scales:Boolean",
        "Continuous": "seaborn._core.scales:Continuous",
        "Nominal": "seaborn._core.scales:Nominal",
        "Temporal": "seaborn._core.scales:Temporal",
        "Move": "seaborn._core.moves:Move",
        "Dodge": "seaborn._core.moves:Dodge",
        "Jitter": "seaborn._core.moves:Jitter",
        "Norm": "seaborn._core.moves:Norm",
        "Shift": "seaborn._core.moves:Shift",
        "Stack": "seaborn._core.moves:Stack",
        "Plot": "seaborn._core.plot:Plot",
        "PROPERTIES": "seaborn._core.properties:PROPERTIES",
        "Property": "seaborn._core.properties:Property",
        "RGBATuple": "seaborn._core.properties:RGBATuple",
        "DashPattern": "seaborn._core.properties:DashPattern",
        "DashPatternWithOffset": "seaborn._core.properties:DashPatternWithOffset",
    }
    if name in _LAZY:
        mod_path, _, attr = _LAZY[name].partition(":")
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)

    raise AttributeError(f"module 'seaborn._core' has no attribute '{name}'")


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
    "GroupBy",
    "Scale",
    "Boolean",
    "Continuous",
    "Nominal",
    "Temporal",
    "Move",
    "Dodge",
    "Jitter",
    "Norm",
    "Shift",
    "Stack",
    "Plot",
    "PROPERTIES",
    "Property",
    "RGBATuple",
    "DashPattern",
    "DashPatternWithOffset",
]
