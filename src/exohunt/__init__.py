"""exohunt: a TESS planet-search pipeline.

Stages (each a module you can swap out):
    ingest   -> download + clean QLP light curves         (exohunt.ingest)
    detrend  -> remove stellar/systematic trends          (exohunt.detrend)
    search   -> periodic transit search (TLS, later GPU)  (exohunt.search)
    vet      -> astrophysical false-positive checks       (week 2+)
    classify -> CNN ranking                               (week 3)
"""

__version__ = "0.1.0"
