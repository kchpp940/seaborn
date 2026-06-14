from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seaborn._core.order import OrderRegistry, categorical_order, _OrderEntry


class TestOrderEntry:

    def test_defaults(self):
        entry = _OrderEntry()
        assert entry.user_order is None
        assert entry.resolved_order is None
        assert entry.is_explicit is False
        assert entry.source is None

    def test_with_user_order(self):
        entry = _OrderEntry(user_order=["a", "b"], is_explicit=True)
        assert entry.user_order == ["a", "b"]
        assert entry.is_explicit is True


class TestCategoricalOrder:

    def test_basic_string_order(self):
        s = pd.Series(["b", "a", "c", "a", "b"])
        result = categorical_order(s)
        assert result == ["b", "a", "c"]

    def test_numeric_order_sorted(self):
        s = pd.Series([3, 1, 2])
        result = categorical_order(s)
        assert result == [1, 2, 3]

    def test_explicit_order(self):
        s = pd.Series(["b", "a", "c"])
        result = categorical_order(s, order=["c", "a", "b"])
        assert result == ["c", "a", "b"]

    def test_categorical_dtype(self):
        s = pd.Categorical(["b", "a", "c"], categories=["c", "b", "a"], ordered=True)
        result = categorical_order(pd.Series(s))
        assert result == ["c", "b", "a"]

    def test_drops_na(self):
        s = pd.Series(["b", None, "a", np.nan])
        result = categorical_order(s)
        assert None not in result
        assert np.nan not in result

    def test_empty_series(self):
        s = pd.Series(dtype=object)
        result = categorical_order(s)
        assert result == []


class TestOrderRegistry:

    def test_basic_register_and_get(self):
        df = pd.DataFrame({"hue": ["b", "a", "c"]})
        registry = OrderRegistry(data=df)
        registry.register("hue")
        result = registry.get("hue")
        assert result == ["b", "a", "c"]

    def test_register_with_explicit_order(self):
        df = pd.DataFrame({"hue": ["b", "a", "c"]})
        registry = OrderRegistry(data=df)
        registry.register("hue", order=["c", "a", "b"])
        result = registry.get("hue")
        assert result == ["c", "a", "b"]

    def test_is_explicit(self):
        registry = OrderRegistry()
        registry.register("hue", order=["a", "b"])
        assert registry.is_explicit("hue") is True

        registry.register("size")
        assert registry.is_explicit("size") is False

    def test_is_explicit_unregistered(self):
        registry = OrderRegistry()
        assert registry.is_explicit("hue") is False

    def test_numeric_order_sorted(self):
        df = pd.DataFrame({"x": [3, 1, 2]})
        registry = OrderRegistry(data=df)
        registry.register("x")
        result = registry.get("x")
        assert result == [1, 2, 3]

    def test_categorical_dtype_order(self):
        cat = pd.Categorical(["b", "a", "c"], categories=["c", "b", "a"])
        df = pd.DataFrame({"hue": cat})
        registry = OrderRegistry(data=df)
        registry.register("hue")
        result = registry.get("hue")
        assert result == ["c", "b", "a"]

    def test_register_with_data_vector(self):
        registry = OrderRegistry()
        data = pd.Series(["b", "a", "c"])
        registry.register("hue", data=data)
        result = registry.get("hue")
        assert result == ["b", "a", "c"]

    def test_register_with_data_and_explicit_order(self):
        registry = OrderRegistry()
        data = pd.Series(["b", "a", "c"])
        registry.register("hue", order=["c", "b", "a"], data=data)
        result = registry.get("hue")
        assert result == ["c", "b", "a"]

    def test_contains(self):
        registry = OrderRegistry()
        assert "hue" not in registry
        registry.register("hue")
        assert "hue" in registry

    def test_getitem(self):
        df = pd.DataFrame({"hue": ["b", "a", "c"]})
        registry = OrderRegistry(data=df)
        registry.register("hue")
        assert registry["hue"] == ["b", "a", "c"]

    def test_getitem_missing_raises(self):
        registry = OrderRegistry()
        with pytest.raises(KeyError):
            registry["hue"]

    def test_invalid_var_name(self):
        registry = OrderRegistry()
        with pytest.raises(ValueError, match="Invalid variable name"):
            registry.register("invalid_var")

    def test_method_chaining(self):
        df = pd.DataFrame({"hue": ["b", "a"], "size": [1, 2]})
        registry = OrderRegistry(data=df)
        result = registry.register("hue").register("size")
        assert result is registry

    def test_update_data(self):
        df1 = pd.DataFrame({"hue": ["b", "a", "c"]})
        df2 = pd.DataFrame({"hue": ["x", "y", "z"]})
        registry = OrderRegistry(data=df1)
        registry.register("hue")
        assert registry["hue"] == ["b", "a", "c"]

        registry.update_data(df2)
        registry.register("size", data=df2["hue"])
        assert registry["size"] == ["x", "y", "z"]

    def test_update_resolved(self):
        registry = OrderRegistry()
        registry.register("hue", order=["a", "b"])
        assert registry["hue"] == ["a", "b"]

        registry.update_resolved("hue", ["b", "a"])
        assert registry["hue"] == ["b", "a"]

    def test_update_resolved_unregistered(self):
        registry = OrderRegistry()
        registry.update_resolved("hue", ["a", "b"])
        assert "hue" not in registry

    def test_resolve_lazy(self):
        df = pd.DataFrame({"hue": ["b", "a", "c"]})
        registry = OrderRegistry(data=df)
        registry.register("hue")

        entry = registry._orders["hue"]
        assert entry.resolved_order is None

        result = registry.get("hue")
        assert result == ["b", "a", "c"]
        assert entry.resolved_order is not None

    def test_get_without_resolve(self):
        registry = OrderRegistry()
        registry.register("hue")
        result = registry.get("hue", resolve=False)
        assert result is None

    def test_var_names_mapping(self):
        df = pd.DataFrame({"species": ["b", "a", "c"]})
        registry = OrderRegistry(data=df, var_names={"hue": "species"})
        registry.register("hue")
        result = registry.get("hue")
        assert result == ["b", "a", "c"]

    def test_repr(self):
        registry = OrderRegistry()
        registry.register("hue", order=["a", "b"])
        registry.resolve("hue", data=pd.Series(["a", "b"]))
        r = repr(registry)
        assert "hue" in r
        assert "explicit" in r

    def test_drops_na_in_resolved_order(self):
        df = pd.DataFrame({"hue": ["b", None, "a"]})
        registry = OrderRegistry(data=df)
        registry.register("hue")
        result = registry.get("hue")
        assert None not in result

    def test_get_unregistered_with_data(self):
        df = pd.DataFrame({"hue": ["b", "a", "c"]})
        registry = OrderRegistry(data=df)
        result = registry.get("hue")
        assert result == ["b", "a", "c"]

    def test_shared_registry_preserves_resolved(self):
        df = pd.DataFrame({"hue": ["b", "a", "c"], "row": [1, 2, 3]})
        registry = OrderRegistry(data=df)
        registry.register("hue")
        hue_order = registry.get("hue")

        row_data = pd.Series([3, 2, 1])
        registry.register("row", data=row_data)
        assert registry.get("hue") == hue_order

    def test_update_var_names(self):
        df = pd.DataFrame({"species": ["b", "a", "c"]})
        registry = OrderRegistry(data=df, var_names={"hue": "species"})
        registry.register("hue")
        result = registry.get("hue")
        assert result == ["b", "a", "c"]

        registry.update_var_names({"size": "weight"})
        registry.register("size", order=["x", "y"])
        assert registry.get("size") == ["x", "y"]
