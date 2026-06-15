"""
Internal core module for seaborn's declarative objects interface.

**Import boundary rules:**

1. **Within ``_core`` itself** (``rules.py``, ``typing.py``, etc.):
   Use **relative imports** (``from .rules import X``). Never import from
   ``seaborn._core`` (the package root) to avoid circular dependencies.

2. **Objects interface layer** (``_marks/*``, ``_stats/*``, ``objects.py``):
   - ``objects.py`` is **exempt** — it re-exports public API classes
     (``Plot``, ``Scale``, ``Move``, …) directly from their defining
     sub-modules for ``from seaborn.objects import …`` usage.
   - ``_marks/*`` and ``_stats/*`` may import from this ``__init__``
     module, but note that some symbols (``Scale``, ``GroupBy``, etc.)
     are lazily imported to avoid circular dependencies during
     ``import seaborn``.

3. **Upper-layer plotter modules** (``_base.py``, ``axisgrid.py``,
   ``categorical.py``, ``utils.py``, etc.):
   Import shared helpers **only** from this ``__init__`` module,
   not from individual ``_core`` sub-modules.

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

# ---------------------------------------------------------------------------
# Symbols below are shared with the objects interface layer (_marks, _stats)
# but are NOT intended for use by upper-layer plotter modules.
#
# They are lazily imported to avoid circular dependencies during package
# initialization (seaborn.palettes -> seaborn.utils -> seaborn._core -> ...).
# ---------------------------------------------------------------------------

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
    from seaborn._core.moves import Move
    from seaborn._core.properties import (
        PROPERTIES,
        Property,
        RGBATuple,
        DashPattern,
        DashPatternWithOffset,
    )


def __getattr__(name: str):
    """Lazy import for objects-interface symbols to avoid circular deps."""
    if name == "GroupBy":
        from seaborn._core.groupby import GroupBy
        return GroupBy
    if name == "Scale":
        from seaborn._core.scales import Scale
        return Scale
    if name == "Boolean":
        from seaborn._core.scales import Boolean
        return Boolean
    if name == "Continuous":
        from seaborn._core.scales import Continuous
        return Continuous
    if name == "Nominal":
        from seaborn._core.scales import Nominal
        return Nominal
    if name == "Temporal":
        from seaborn._core.scales import Temporal
        return Temporal
    if name == "Move":
        from seaborn._core.moves import Move
        return Move
    if name == "PROPERTIES":
        from seaborn._core.properties import PROPERTIES
        return PROPERTIES
    if name == "Property":
        from seaborn._core.properties import Property
        return Property
    if name == "RGBATuple":
        from seaborn._core.properties import RGBATuple
        return RGBATuple
    if name == "DashPattern":
        from seaborn._core.properties import DashPattern
        return DashPattern
    if name == "DashPatternWithOffset":
        from seaborn._core.properties import DashPatternWithOffset
        return DashPatternWithOffset

    raise AttributeError(f"module 'seaborn._core' has no attribute '{name}'")


__all__ = [
    # ------------------------------------------------------------------
    # Stable API for upper-layer plotter modules (_base, axisgrid, etc.)
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
    # Extended API for objects interface (_marks, _stats) only
    # These are lazily imported — see __getattr__ above.
    # ------------------------------------------------------------------
    "GroupBy",
    "Scale",
    "Boolean",
    "Continuous",
    "Nominal",
    "Temporal",
    "Move",
    "PROPERTIES",
    "Property",
    "RGBATuple",
    "DashPattern",
    "DashPatternWithOffset",
]
