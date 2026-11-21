"""Stage 3: transit search. Each searcher returns `Detection`s.

`tls` is the CPU reference. `bls` is our own Box Least Squares with a CPU
(numba) and a CUDA backend, same `search(lc, **kw) -> Detection` signature.
"""

from .base import Detection, iterative_search
from .bls import bls_search
from .tls import tls_search

SEARCHERS = {"tls": tls_search, "bls": bls_search}

__all__ = ["Detection", "iterative_search", "tls_search", "bls_search", "SEARCHERS"]
