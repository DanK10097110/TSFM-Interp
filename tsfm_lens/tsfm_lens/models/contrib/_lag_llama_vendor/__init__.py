"""Vendored Lag-Llama model source (Apache-2.0), not a contrib adapter itself.

`models/contrib/__init__.py::discover_contrib_adapters` only globs the
IMMEDIATE `contrib/*.py` files for `ADAPTER_NAME`/`ADAPTER_CLASS` literals, so
this subpackage (with neither) is invisible to that scan and never appears in
`--doctor`'s discovery-error list. See `model.py` for provenance and the
license header, kept verbatim on each vendored file per Apache-2.0 sec 4(c).
"""
