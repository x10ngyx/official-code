from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Iterable

from .models import ManualComponent, ProfileCase

ProfileItem = ProfileCase | ManualComponent


def _load_module(reference: str) -> ModuleType:
    path = Path(reference)
    if path.suffix == ".py" or path.exists():
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError(f"Adapter path is not a file: {resolved}")
        module_name = f"calflops_eval_adapter_{abs(hash(str(resolved)))}"
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load adapter module from {resolved}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(reference)


def load_profile_items(specification: str) -> list[ProfileItem]:
    if ":" not in specification:
        raise ValueError("Adapter must use MODULE_OR_FILE.py:FACTORY format")
    module_reference, factory_name = specification.rsplit(":", 1)
    module = _load_module(module_reference)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError(f"Adapter factory is not callable: {factory_name}")
    produced = factory()
    if isinstance(produced, (ProfileCase, ManualComponent)):
        items = [produced]
    elif isinstance(produced, Iterable):
        items = list(produced)
    else:
        raise TypeError("Adapter factory must return a profile item or iterable of items")
    if not items:
        raise ValueError("Adapter returned no profile items")
    for item in items:
        if not isinstance(item, (ProfileCase, ManualComponent)):
            raise TypeError(f"Unsupported profile item: {type(item).__name__}")
    names = [item.name for item in items]
    if any(not name for name in names):
        raise ValueError("Profile item names must be non-empty")
    if len(names) != len(set(names)):
        raise ValueError("Profile item names must be unique")
    return items
