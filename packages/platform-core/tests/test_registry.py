import pytest

from platform_core.registry import Registry


def test_register_and_get():
    registry: Registry[str] = Registry(kind="entry")
    registry.register("a", "value-a")

    assert registry.get("a") == "value-a"
    assert "a" in registry
    assert "b" not in registry


def test_register_rejects_duplicate_key():
    registry: Registry[str] = Registry(kind="widget")
    registry.register("a", "value-a")

    with pytest.raises(ValueError, match="widget 'a' already registered"):
        registry.register("a", "value-a-again")


def test_all_returns_a_copy():
    registry: Registry[str] = Registry()
    registry.register("a", "value-a")

    snapshot = registry.all()
    snapshot["b"] = "leaked"

    assert registry.all() == {"a": "value-a"}


def test_clear_empties_the_registry():
    registry: Registry[str] = Registry()
    registry.register("a", "value-a")

    registry.clear()

    assert registry.all() == {}
