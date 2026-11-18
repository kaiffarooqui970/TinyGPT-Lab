"""YAML config loading into typed, validated dataclasses.

Every experiment is described by a dataclass. `from_dict` builds one from a plain
dict (parsed YAML), recursing into nested dataclasses and checking types, so that a
typo such as `stpes: 100` fails loudly instead of being silently ignored.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a config dict does not match its dataclass schema."""


def _convert(value: Any, tp: Any, path: str) -> Any:
    """Check and convert `value` to type `tp`; `path` is used in error messages."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if tp is Any:
        return value
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping for {tp.__name__}, got {value!r}")
        return from_dict(tp, value, _path=path)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in (a for a in args if a is not type(None)):
            try:
                return _convert(value, arg, path)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(f"{path}: {value!r} matches none of {args}: {errors}")
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected list, got {value!r}")
        (inner,) = args or (Any,)
        return [_convert(v, inner, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected sequence, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(
            _convert(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args, strict=True))
        )
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping, got {value!r}")
        kt, vt = args or (Any, Any)
        return {_convert(k, kt, path): _convert(v, vt, f"{path}.{k}") for k, v in value.items()}
    if tp is float:
        # YAML parses `1` as int and `1e-3` (no dot) as str; accept both as floats.
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected float, got bool")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ConfigError(f"{path}: expected float, got {value!r}")
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected int, got {value!r}")
        return value
    if isinstance(tp, type):
        if not isinstance(value, tp):
            raise ConfigError(f"{path}: expected {tp.__name__}, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported annotation {tp!r}")


def from_dict(cls: type[T], data: dict[str, Any], _path: str = "") -> T:
    """Build dataclass `cls` from `data`, validating keys and types recursively.

    Raises:
        ConfigError: on unknown keys, missing required fields, or type mismatches.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"{_path or cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        sub = f"{_path}.{name}" if _path else name
        if name in data:
            kwargs[name] = _convert(data[name], hints[name], sub)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{sub}: missing required field")
    return cls(**kwargs)


def load_config(path: str | Path, cls: type[T]) -> T:
    """Load a YAML file into dataclass `cls`."""
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return from_dict(cls, data)


def to_dict(cfg: Any) -> dict[str, Any]:
    """Convert a (nested) dataclass to a YAML-serialisable dict; tuples become lists."""

    def clean(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v

    return clean(dataclasses.asdict(cfg))


def save_config(cfg: Any, path: str | Path) -> None:
    """Write a dataclass config to YAML."""
    with open(path, "w") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)
