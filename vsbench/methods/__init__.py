"""The method registry: ``@register(name)`` on a class, ``make_method(name, **params)``.

A student adds a method with one class and one ``@register("my_method")`` line; the
built-in methods are registered when this package is imported. Nothing under
``vsbench/methods/`` imports the episode loop, a renderer or a scene (a test checks the
import graph).
"""

from __future__ import annotations

from typing import Callable, TypeVar

from vsbench.methods.base import (
    DEPTH_MODES,
    INFO_KEYS,
    DepthMode,
    VSMethod,
    check_method,
    normalise_output,
)
from vsbench.methods.masking import mask_matches, takes_goal_mask

_REGISTRY: dict[str, type] = {}
T = TypeVar("T", bound=type)


def register(name: str) -> Callable[[T], T]:
    """Class decorator: make ``make_method(name)`` build this class and set its ``name``."""
    if not isinstance(name, str) or not name:
        raise ValueError("a method name must be a non-empty string")

    def deco(cls: T) -> T:
        have = _REGISTRY.get(name)
        if have is not None and have is not cls and have.__qualname__ != cls.__qualname__:
            raise ValueError(f"method {name!r} is already registered by {have.__module__}.{have.__qualname__}")
        own = cls.__dict__.get("name")
        if own is not None and own != name:
            raise ValueError(f"class {cls.__name__} says name={own!r} but is registered as {name!r}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def available_methods() -> list[str]:
    return sorted(_REGISTRY)


def method_class(name: str) -> type:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; known: {available_methods()}") from None


def make_method(name: str, **params) -> VSMethod:
    """Build the registered method with its constructor parameters and check the contract.

    The instance gets a ``params`` attribute holding the constructor arguments (unless
    the class defines one), so the episode meta can record them.
    """
    m = method_class(name)(**params)
    check_method(m)
    if not hasattr(m, "params"):
        m.params = dict(params)
    return m


# built-in methods register themselves on import (after ``register`` exists)
# (vssim_arms: the copied vs-sim arms as methods, PORT_PLAN U6b; their bench and weights
# load at configure, so this import stays light)
from vsbench.methods import (  # noqa: E402,F401
    feature_ibvs,
    oracle_pbvs,
    true_point_ibvs,
    vssim_arms,
)

__all__ = [
    "DEPTH_MODES",
    "INFO_KEYS",
    "DepthMode",
    "VSMethod",
    "available_methods",
    "check_method",
    "make_method",
    "mask_matches",
    "method_class",
    "normalise_output",
    "register",
    "takes_goal_mask",
]
