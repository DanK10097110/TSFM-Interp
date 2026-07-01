"""Minimal name-to-factory registry shared by every plugin category.

Generators, corruptions, and sources all register themselves here so the
declarative config can refer to them by string. This keeps the assembly layer
unaware of concrete implementations and makes adding a new strategy a one-line
decorator rather than an edit to the builder.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Holds named entries of a single plugin category."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator that records an implementation under a unique name."""

        def wrap(obj: T) -> T:
            if name in self._entries:
                raise ValueError(f"{self._kind} '{name}' already registered")
            self._entries[name] = obj
            return obj

        return wrap

    def get(self, name: str) -> T:
        if name not in self._entries:
            raise KeyError(f"unknown {self._kind} '{name}'; have {sorted(self._entries)}")
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)


GENERATORS: Registry = Registry("generator")
CORRUPTIONS: Registry = Registry("corruption")
SOURCES: Registry = Registry("source")
