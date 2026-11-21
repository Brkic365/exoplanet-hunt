"""Box Least Squares written from scratch, with a CPU (numba) and a CUDA kernel.

Both backends run the same algorithm, so they return the same periodogram:

1. Phase-fold the light curve at a trial period into `nbins` bins of width
   `dur_min / oversample` (weighted sums of w = 1/sigma^2 and w*y, where y is the
   flux minus its weighted mean).
2. Prefix-sum the bins, so any box (start bin j, length n bins, wrapping around
   phase 1 -> 0) costs O(1).
3. For every box with a duration from the grid, the dip statistic is
   delta-chi2 = s^2 / (r (1 - r)), with r = sum(w in box) / W and
   s = sum(w y in box) / W. Only dips (s < 0) count.
4. Keep the best box per period. Host code turns the periodogram into an SDE
   the same way TLS does, so SDE thresholds are comparable.

The CUDA kernel runs one thread block per (period, star): the block bins the
star's points into shared memory with atomics, scans the bins cooperatively,
then each thread scores a slice of start phases and the block reduces to the
best box. Many stars go in one launch (`bls_batch`), which is where the GPU wins.
Without a GPU the kernel can be run in numba's CUDA simulator
(NUMBA_ENABLE_CUDASIM=1) to check it against the CPU path; that is what the tests do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import cuda, float32, int32, njit, prange

from ..ingest import LightCurve
from .base import Detection

# durations searched [days]; a box must also be < MAX_Q of the period
DURATIONS_HR = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
MAX_Q = 0.12
NBINS_MAX = 4096  # shared-memory budget per block (2 x float32 x 4097 ~ 33 kB)
THREADS = 128


@dataclass
class Grid:
    periods: np.ndarray  # [n_p] days
    nbins: np.ndarray  # [n_p] int32 bins per period
    dur_bins: np.ndarray  # [n_p, n_d] int32 box lengths in bins (0 = skip)


def make_grid(baseline: float, period_min: float = 0.6, period_max: float | None = None,
              oversampling_factor: int = 3, bin_oversample: int = 3, durations_hr=DURATIONS_HR,
              min_transits: int = 2) -> Grid:
    """Period grid identical to TLS's (Ofir 2014, solar star) so the two searches
    see the same trial periods; boxes from `durations_hr` that fit inside MAX_Q."""
    from transitleastsquares import period_grid

    if period_max is None:
        period_max = baseline / min_transits
    periods = np.sort(period_grid(1.0, 1.0, baseline, period_min, period_max, oversampling_factor))
    durs = np.asarray(durations_hr, float) / 24.0
    nbins = np.empty(len(periods), np.int32)
    dur_bins = np.zeros((len(periods), len(durs)), np.int32)
    for i, p in enumerate(periods):
        ok = durs < MAX_Q * p
        dmin = durs[ok].min() if ok.any() else durs.min()
        nb = int(min(NBINS_MAX, max(16, math.ceil(bin_oversample * p / dmin))))
        nbins[i] = nb
        bw = p / nb
        dur_bins[i, ok] = np.maximum(1, np.round(durs[ok] / bw)).astype(np.int32)
    return Grid(periods, nbins, dur_bins)


def _prep(lc: LightCurve) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Centered flux and normalised weights (sum w = 1), times relative to t_ref."""
    t_ref = float(lc.time.min())
    w = 1.0 / np.square(np.asarray(lc.flux_err, float))
    w[~np.isfinite(w)] = 0.0
    w /= w.sum()
    y = lc.flux - np.sum(w * lc.flux)
    return (lc.time - t_ref).astype(np.float64), y.astype(np.float32), w.astype(np.float32), t_ref


# ----------------------------------------------------------------------------- CPU


@njit(cache=True, fastmath=True)
def _bls_one_period(t, y, w, period, nb, dbins, bw, bs):
    bw[: nb + 1] = 0.0
    bs[: nb + 1] = 0.0
    inv = 1.0 / period
    for i in range(t.size):
        ph = t[i] * inv
        ph -= math.floor(ph)
        b = int(ph * nb)
        if b >= nb:
            b = nb - 1
        bw[b + 1] += w[i]
        bs[b + 1] += w[i] * y[i]
    for b in range(nb):  # prefix sums, cum[0] = 0
        bw[b + 1] += bw[b]
        bs[b + 1] += bs[b]
    best, bj, bk = 0.0, 0, 0
    for k in range(dbins.size):
        n = dbins[k]
        if n <= 0 or n >= nb:
            continue
        for j in range(nb):
            e = j + n
            if e <= nb:
                r = bw[e] - bw[j]
                s = bs[e] - bs[j]
            else:
                r = bw[nb] - bw[j] + bw[e - nb]
                s = bs[nb] - bs[j] + bs[e - nb]
            if s < 0.0 and r > 0.0 and r < 1.0:
                p = s * s / (r * (1.0 - r))
                if p > best:
                    best, bj, bk = p, j, k
    return best, bj, bk


@njit(parallel=True, cache=True)
def _bls_cpu(t, y, w, periods, nbins, dur_bins):
    n_p = periods.size
    power = np.zeros(n_p)
    jbest = np.zeros(n_p, np.int32)
    kbest = np.zeros(n_p, np.int32)
    for i in prange(n_p):
        bw = np.empty(nbins[i] + 1)
        bs = np.empty(nbins[i] + 1)
        p, j, k = _bls_one_period(t, y, w, periods[i], nbins[i], dur_bins[i], bw, bs)
        power[i], jbest[i], kbest[i] = p, j, k
    return power, jbest, kbest


def bls_periodogram_cpu(lc: LightCurve, grid: Grid):
    t, y, w, t_ref = _prep(lc)
    p, j, k = _bls_cpu(t, y.astype(np.float64), w.astype(np.float64), grid.periods, grid.nbins, grid.dur_bins)
    return p, j, k, t_ref


# ----------------------------------------------------------------------------- CUDA

_kernel_cache: dict = {}


def _get_kernel(threads: int = THREADS):
    """Build (once) the CUDA kernel for a block size. `cuda` is a module global
    so numba's simulator can swap it in."""
    if threads in _kernel_cache:
        return _kernel_cache[threads]
    NB1 = NBINS_MAX + 1
    TPB = threads

    @cuda.jit
    def bls_kernel(t, y, w, offsets, periods, nbins, dur_bins, out_power, out_j, out_k):
        ip = cuda.blockIdx.x  # period
        istar = cuda.blockIdx.y  # star
        tid = cuda.threadIdx.x
        sw = cuda.shared.array(NB1, float32)
        ss = cuda.shared.array(NB1, float32)
        chunk_w = cuda.shared.array(TPB, float32)
        chunk_s = cuda.shared.array(TPB, float32)
        red_p = cuda.shared.array(TPB, float32)
        red_i = cuda.shared.array(TPB, int32)

        nb = nbins[ip]
        period = periods[ip]
        inv = 1.0 / period
        i0 = offsets[istar]
        i1 = offsets[istar + 1]

        for b in range(tid, nb + 1, TPB):
            sw[b] = 0.0
            ss[b] = 0.0
        cuda.syncthreads()

        # 1. fold + bin (shared-memory atomics); bin b lands in slot b+1
        for i in range(i0 + tid, i1, TPB):
            ph = t[i] * inv
            ph -= math.floor(ph)
            b = int32(ph * nb)
            if b >= nb:
                b = nb - 1
            cuda.atomic.add(sw, b + 1, w[i])
            cuda.atomic.add(ss, b + 1, w[i] * y[i])
        cuda.syncthreads()

        # 2. block prefix sum over slots 1..nb: each thread scans a contiguous
        # chunk, thread 0 scans the chunk totals, then offsets are added back.
        per = (nb + TPB - 1) // TPB
        lo = 1 + tid * per
        hi = min(lo + per, nb + 1)
        accw = float32(0.0)
        accs = float32(0.0)
        for b in range(lo, hi):
            accw += sw[b]
            sw[b] = accw
            accs += ss[b]
            ss[b] = accs
        chunk_w[tid] = accw
        chunk_s[tid] = accs
        cuda.syncthreads()
        if tid == 0:
            cw = float32(0.0)
            cs = float32(0.0)
            for c in range(TPB):
                tw = chunk_w[c]
                ts = chunk_s[c]
                chunk_w[c] = cw
                chunk_s[c] = cs
                cw += tw
                cs += ts
        cuda.syncthreads()
        offw = chunk_w[tid]
        offs = chunk_s[tid]
        for b in range(lo, hi):
            sw[b] += offw
            ss[b] += offs
        cuda.syncthreads()

        # 3. score boxes: thread handles start bins j = tid, tid+TPB, ...
        best = float32(0.0)
        bidx = int32(0)
        totw = sw[nb]
        tots = ss[nb]
        nd = dur_bins.shape[1]
        for k in range(nd):
            n = dur_bins[ip, k]
            if n <= 0 or n >= nb:
                continue
            for j in range(tid, nb, TPB):
                e = j + n
                if e <= nb:
                    r = sw[e] - sw[j]
                    s = ss[e] - ss[j]
                else:
                    r = totw - sw[j] + sw[e - nb]
                    s = tots - ss[j] + ss[e - nb]
                if s < 0.0 and r > 0.0 and r < 1.0:
                    p = s * s / (r * (1.0 - r))
                    if p > best:
                        best = p
                        bidx = k * NB1 + j
        red_p[tid] = best
        red_i[tid] = bidx
        cuda.syncthreads()

        # 4. block reduction to the best box (TPB is a power of two)
        step = TPB // 2
        while step > 0:
            if tid < step and red_p[tid + step] > red_p[tid]:
                red_p[tid] = red_p[tid + step]
                red_i[tid] = red_i[tid + step]
            cuda.syncthreads()
            step //= 2
        if tid == 0:
            o = istar * periods.shape[0] + ip
            out_power[o] = red_p[0]
            out_j[o] = red_i[0] % NB1
            out_k[o] = red_i[0] // NB1

    _kernel_cache[threads] = bls_kernel
    return bls_kernel


def cuda_available() -> bool:
    try:
        return cuda.is_available()
    except Exception:
        return False


def bls_periodogram_cuda(lcs: list[LightCurve], grid: Grid, threads: int = THREADS):
    """Periodograms for many stars in one kernel launch. Returns arrays shaped
    [n_star, n_period] plus each star's t_ref."""
    preps = [_prep(lc) for lc in lcs]
    offsets = np.zeros(len(lcs) + 1, np.int64)
    offsets[1:] = np.cumsum([len(p[0]) for p in preps])
    t = np.concatenate([p[0] for p in preps]).astype(np.float32)  # rel. time, <~30 d: float32 is enough
    y = np.concatenate([p[1] for p in preps]).astype(np.float32)
    w = np.concatenate([p[2] for p in preps]).astype(np.float32)
    n_s, n_p = len(lcs), len(grid.periods)
    d = {k: cuda.to_device(v) for k, v in dict(
        t=t, y=y, w=w, offsets=offsets, periods=grid.periods.astype(np.float32), nbins=grid.nbins,
        dur_bins=grid.dur_bins).items()}
    out_p = cuda.device_array(n_s * n_p, np.float32)
    out_j = cuda.device_array(n_s * n_p, np.int32)
    out_k = cuda.device_array(n_s * n_p, np.int32)
    kern = _get_kernel(threads)
    kern[(n_p, n_s), threads](d["t"], d["y"], d["w"], d["offsets"], d["periods"], d["nbins"], d["dur_bins"],
                               out_p, out_j, out_k)
    cuda.synchronize()
    shape = (n_s, n_p)
    return (out_p.copy_to_host().reshape(shape).astype(np.float64), out_j.copy_to_host().reshape(shape),
            out_k.copy_to_host().reshape(shape), [p[3] for p in preps])


# ----------------------------------------------------------------------------- results


def sde(power: np.ndarray, kernel: int = 101) -> tuple[float, np.ndarray]:
    """SDE of a BLS periodogram: sqrt(delta-chi2) with a running-median trend
    removed, then (peak - mean) / std. Same idea as TLS's detrended SDE."""
    from scipy.ndimage import median_filter

    a = np.sqrt(np.maximum(power, 0.0))
    k = min(kernel, (len(a) // 2) * 2 - 1)
    a = a - median_filter(a, size=max(k, 3), mode="nearest")
    sd = np.std(a)
    if not np.isfinite(sd) or sd <= 0:
        return 0.0, a
    return float((a.max() - a.mean()) / sd), a


def to_detection(lc: LightCurve, grid: Grid, power, jbest, kbest, t_ref, method="bls") -> Detection | None:
    if not np.any(power > 0):
        return None
    s, detr = sde(power)
    i = int(np.argmax(detr))
    P = float(grid.periods[i])
    nb = int(grid.nbins[i])
    n = int(grid.dur_bins[i, kbest[i]])
    bwidth = P / nb
    dur = n * bwidth
    t0 = t_ref + (jbest[i] + 0.5 * n) * bwidth
    # move t0 to the first epoch inside the data
    t0 = t0 + P * math.ceil((lc.time.min() - t0) / P)
    ph = ((lc.time - t0 + 0.5 * P) % P) - 0.5 * P
    intr = np.abs(ph) < 0.5 * dur
    depth = float(np.median(lc.flux[~intr]) - np.mean(lc.flux[intr])) if intr.any() else float("nan")
    epochs = np.unique(np.round((lc.time[intr] - t0) / P)) if intr.any() else []
    noise = 1.4826 * np.median(np.abs(lc.flux - np.median(lc.flux)))
    snr = depth / noise * math.sqrt(max(intr.sum(), 1)) if noise > 0 else float("nan")
    return Detection(tic=lc.tic, period=P, t0=float(t0), duration=float(dur), depth=depth, sde=s,
                     snr=float(snr), n_transits=int(len(epochs)), method=method,
                     extra={"delta_chi2": float(power[i])})


def bls_search(lc: LightCurve, period_min: float = 0.6, period_max: float | None = None,
               backend: str = "cpu", grid: Grid | None = None, min_transits: int = 2, **_) -> Detection | None:
    """Single-star BLS with the same signature as `tls_search`."""
    baseline = float(lc.time.max() - lc.time.min())
    grid = grid or make_grid(baseline, period_min, period_max, min_transits=min_transits)
    if len(grid.periods) == 0:
        return None
    if backend == "cuda":
        p, j, k, tr = bls_periodogram_cuda([lc], grid)
        return to_detection(lc, grid, p[0], j[0], k[0], tr[0], method="bls_cuda")
    p, j, k, tr = bls_periodogram_cpu(lc, grid)
    return to_detection(lc, grid, p, j, k, tr, method="bls")


def bls_batch(lcs: list[LightCurve], grid: Grid, backend: str = "cuda") -> list[Detection | None]:
    """Best signal for each star; all stars share `grid` (same sector baseline)."""
    if backend == "cuda":
        p, j, k, tr = bls_periodogram_cuda(lcs, grid)
        return [to_detection(lc, grid, p[i], j[i], k[i], tr[i], method="bls_cuda") for i, lc in enumerate(lcs)]
    out = []
    for lc in lcs:
        p, j, k, tr = bls_periodogram_cpu(lc, grid)
        out.append(to_detection(lc, grid, p, j, k, tr))
    return out
