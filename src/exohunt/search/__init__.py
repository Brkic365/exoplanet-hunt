"""Stage 3: transit search. Each searcher returns `Detection`s.

`tls` is the CPU reference. A GPU BLS (week 2) implements the same
`search(lc, **kw) -> list[Detection]` signature so it can be swapped in and
benchmarked against TLS.
"""

from .base import Detection, iterative_search
from .tls import tls_search

SEARCHERS = {"tls": tls_search}

__all__ = ["Detection", "iterative_search", "tls_search", "SEARCHERS"]
