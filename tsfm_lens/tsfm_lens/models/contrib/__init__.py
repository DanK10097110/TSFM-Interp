"""A home for contributed adapters (`ROADMAP.md` sec 34.6 Item E1).

Drop one file in this directory and edit nothing else. This module scans its
own directory for files declaring two module-level string literals --
`ADAPTER_NAME` (the registry key) and `ADAPTER_CLASS` (the class to
instantiate) -- and registers `name -> (module_path, class_name)` **without
importing the module**. `models/__init__.py::build_adapter` imports the
module on first actual use, so a contrib file's own heavy imports (a model
library, a tokenizer package) never run at `import tsfm_lens.models` time,
and a contrib file that is entirely broken (a bad import, a syntax error in
a function body the scan never looks at) still registers its name and fails
with a clear, load-time error rather than taking the whole registry down at
import time (`CLAUDE.md` sec 2.5 -- silent skipping is forbidden here, and a
crash at `import tsfm_lens.models` would be worse: the model would simply be
absent with no explanation, for every model in the registry, not just this
one).

`ADAPTER_NAME`/`ADAPTER_CLASS` are read via `ast.parse`, never `exec`/
`importlib` -- the module's *body* never runs during discovery.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CONTRIB_DIR = Path(__file__).resolve().parent


def _literal_str_assignment(tree: ast.Module, name: str) -> Optional[str]:
    """The value of a module-level `name = "..."` assignment, or None.

    Only a bare `ast.Constant` string is accepted -- an f-string, a
    concatenation, or a reference to another name would need real evaluation
    to resolve, which is exactly the executing-the-module step this function
    exists to avoid.
    """
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Name) and target.id == name
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                return node.value.value
    return None


def discover_contrib_adapters() -> Tuple[Dict[str, Tuple[str, str]], List[dict]]:
    """Scan `contrib/*.py` (skipping `__init__.py`) for `ADAPTER_NAME`/`ADAPTER_CLASS`.

    Returns `(registry, errors)`: `registry` maps the declared adapter name to
    `(dotted_module_path, class_name)`, ready for `importlib.import_module` +
    `getattr` at actual `build_adapter` time; `errors` is a list of
    `{"file": ..., "error": ...}` for every file that failed to yield a usable
    registration -- a missing literal, an unparseable file, or (checked by
    the caller, `models/__init__.py`, which knows the built-in names) a name
    collision. Never raises: a broken contrib file is data about that file,
    not a reason to fail every other one.
    """
    registry: Dict[str, Tuple[str, str]] = {}
    errors: List[dict] = []
    for path in sorted(CONTRIB_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            errors.append({"file": path.name, "error": f"SyntaxError: {exc}"})
            continue
        adapter_name = _literal_str_assignment(tree, "ADAPTER_NAME")
        adapter_class = _literal_str_assignment(tree, "ADAPTER_CLASS")
        if not adapter_name:
            errors.append({"file": path.name,
                           "error": "no module-level ADAPTER_NAME = \"...\" string literal"})
            continue
        if not adapter_class:
            errors.append({"file": path.name,
                           "error": "no module-level ADAPTER_CLASS = \"...\" string literal"})
            continue
        if adapter_name in registry:
            other_module, _ = registry[adapter_name]
            errors.append({"file": path.name,
                           "error": f"ADAPTER_NAME '{adapter_name}' already registered by "
                                    f"{other_module} -- rename one of the two"})
            continue
        registry[adapter_name] = (f"tsfm_lens.models.contrib.{path.stem}", adapter_class)
    return registry, errors


def import_contrib_class(module_path: str, class_name: str) -> type:
    """Actually import a contrib module and fetch its adapter class.

    This is the one place a contrib file's own top-level code (and any heavy
    library it imports) actually runs -- deliberately deferred here, away
    from `discover_contrib_adapters`, so a broken contrib file only breaks
    itself, at the moment someone asks for it, not the whole registry at
    startup.
    """
    import importlib

    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        raise AttributeError(f"{module_path} has no class '{class_name}' "
                             f"(declared as ADAPTER_CLASS)")
    return getattr(module, class_name)
