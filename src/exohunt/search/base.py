from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ..ingest import LightCurve


@dataclass
class Detection:
    tic: int
    period: float  # days
    t0: float  # BTJD, mid-transit
    duration: float  # days
    depth: float  # fractional (1 - flux in transit)
    sde: float  # signal detection efficiency
    snr: float
    n_transits: int
    rank: int = 1  # 1 = strongest signal in this star
    method: str = "tls"
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = asdict(self)
        extra = row.pop("extra")
        row.update({f"x_{k}": v for k, v in extra.items()})
        return row


def mask_transits(lc: LightCurve, det: Detection, width: float = 2.0) -> LightCurve:
    """Remove points within `width` x duration of each transit of `det`."""
    phase = ((lc.time - det.t0 + 0.5 * det.period) % det.period) - 0.5 * det.period
    keep = np.abs(phase) > 0.5 * width * det.duration
    return lc.subset(keep)


def iterative_search(
    lc: LightCurve,
    search: Callable[..., Detection | None],
    max_signals: int = 2,
    min_sde: float = 7.0,
    **kw,
) -> list[Detection]:
    """Find up to `max_signals` signals, masking each before searching again."""
    out: list[Detection] = []
    cur = lc
    for rank in range(1, max_signals + 1):
        det = search(cur, **kw)
        if det is None:
            break
        det.rank = rank
        out.append(det)
        if det.sde < min_sde:
            break
        cur = mask_transits(cur, det)
        if len(cur) < 100:
            break
    return out
