from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import uniform_filter as _uniform_filter
except Exception:  # pragma: no cover - optional dependency fallback
    _uniform_filter = None


@dataclass
class ConfigState:
    key: str
    kernel: int
    gx: int
    gy: int
    xx: np.ndarray
    yy: np.ndarray
    frame_gain: List[float]
    frame_mgain: List[int]
    table_sum: np.ndarray
    table_count: np.ndarray


def parse_int_csv(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        value = int(float(tok))
        if value <= 0:
            continue
        out.append(value)
    seen = set()
    dedup: List[int] = []
    for value in out:
        if value in seen:
            continue
        seen.add(value)
        dedup.append(value)
    return dedup


def parse_grid_specs(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for tok in str(text).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if "x" in tok:
            a, b = tok.split("x", 1)
        elif ":" in tok:
            a, b = tok.split(":", 1)
        else:
            a = b = tok
        gx = int(float(a))
        gy = int(float(b))
        if gx <= 1 or gy <= 1:
            continue
        out.append((gx, gy))
    seen = set()
    dedup: List[Tuple[int, int]] = []
    for grid in out:
        if grid in seen:
            continue
        seen.add(grid)
        dedup.append(grid)
    return dedup


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    if values.size == 1:
        return float(values[0])
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights)
    if cdf[-1] <= 0:
        return float(np.quantile(values, q))
    cdf = cdf / cdf[-1]
    return float(np.interp(float(q), cdf, sorted_values))


def _quant_dict(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "q10": float("nan"),
            "q90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    q10, q50, q90 = np.quantile(arr, [0.10, 0.50, 0.90])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(q50),
        "q10": float(q10),
        "q90": float(q90),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def _box_mean_fallback(arr: np.ndarray, k: int) -> np.ndarray:
    radius = int(k) // 2
    padded = np.pad(arr, ((radius, radius), (radius, radius)), mode="edge")
    csum = np.cumsum(np.cumsum(padded, axis=0, dtype=np.float64), axis=1, dtype=np.float64)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    kk = int(k)
    out = csum[kk:, kk:] - csum[:-kk, kk:] - csum[kk:, :-kk] + csum[:-kk, :-kk]
    return out / float(kk * kk)


def box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    if _uniform_filter is not None:
        return _uniform_filter(arr, size=int(k), mode="nearest")
    return _box_mean_fallback(arr, int(k))


def frame_thresholds_from_sample(
    sample_vals: np.ndarray,
    clip_sigma_low: float,
    clip_sigma_high: float,
) -> Tuple[float, float, float, float]:
    ok = sample_vals[np.isfinite(sample_vals)]
    if ok.size < 32:
        mu = float(np.nanmean(ok)) if ok.size else 0.0
        sd = float(np.nanstd(ok)) if ok.size else 1.0
        sd = max(sd, 1e-6)
        return mu, sd, mu - clip_sigma_low * sd, mu + clip_sigma_high * sd
    med = float(np.median(ok))
    mad = float(np.median(np.abs(ok - med)))
    sig = max(1e-6, 1.4826 * mad)
    lo = med - float(clip_sigma_low) * sig
    hi = med + float(clip_sigma_high) * sig
    return med, sig, lo, hi


def make_states(kernels: Sequence[int], grids: Sequence[Tuple[int, int]], h: int, w: int) -> List[ConfigState]:
    states: List[ConfigState] = []
    seen = set()
    min_dim = max(1, min(int(h), int(w)))
    max_kernel = max(1, min_dim if min_dim % 2 == 1 else min_dim - 1)
    for kernel in kernels:
        kernel_eff = max(1, int(kernel))
        if kernel_eff % 2 == 0:
            kernel_eff += 1
        kernel_eff = min(kernel_eff, max_kernel)
        for gx, gy in grids:
            gx_eff = max(2, min(int(gx), int(w)))
            gy_eff = max(2, min(int(gy), int(h)))
            key = (kernel_eff, gx_eff, gy_eff)
            if key in seen:
                continue
            seen.add(key)
            xs = np.linspace(0, w - 1, gx_eff, dtype=np.int32)
            ys = np.linspace(0, h - 1, gy_eff, dtype=np.int32)
            xx, yy = np.meshgrid(xs, ys, indexing="xy")
            states.append(
                ConfigState(
                    key=f"k{kernel_eff}_g{gx_eff}x{gy_eff}",
                    kernel=kernel_eff,
                    gx=gx_eff,
                    gy=gy_eff,
                    xx=xx,
                    yy=yy,
                    frame_gain=[],
                    frame_mgain=[],
                    table_sum=np.zeros((int(gy), int(gx)), dtype=np.float64),
                    table_count=np.zeros((int(gy), int(gx)), dtype=np.int32),
                )
            )
    return states


def _estimate_from_array(
    stack: np.ndarray,
    *,
    kernels: Sequence[int],
    grids: Sequence[Tuple[int, int]],
    sample_pixels_for_clip: int,
    clip_sigma_low: float,
    clip_sigma_high: float,
    min_local_mean: float,
    min_local_var: float,
    min_finite_frac: float,
    max_hot_frac: float,
    gain_min: float,
    gain_max: float,
    seed: int,
) -> Tuple[List[ConfigState], List[int], List[Dict[str, float]]]:
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise ValueError(f"Expected 2D/3D stack, got shape={tuple(stack.shape)}")

    n_frames, h, w = stack.shape
    states = make_states(kernels=kernels, grids=grids, h=h, w=w)
    rng = np.random.default_rng(int(seed))
    sy = rng.integers(0, h, size=max(1024, int(sample_pixels_for_clip)), dtype=np.int32)
    sx = rng.integers(0, w, size=max(1024, int(sample_pixels_for_clip)), dtype=np.int32)

    clip_records: List[Dict[str, float]] = []
    frame_numbers: List[int] = []
    kernels_sorted = sorted({int(state.kernel) for state in states})
    by_kernel: Dict[int, List[ConfigState]] = {}
    for state in states:
        by_kernel.setdefault(int(state.kernel), []).append(state)

    for frame_idx in range(int(n_frames)):
        counts = np.asarray(stack[frame_idx], dtype=np.float64)
        finite = np.isfinite(counts)
        if not np.any(finite):
            for state in states:
                state.frame_gain.append(float("nan"))
                state.frame_mgain.append(0)
            continue

        sample_vals = counts[sy, sx]
        med, sigma, lo, hi = frame_thresholds_from_sample(
            sample_vals=sample_vals,
            clip_sigma_low=float(clip_sigma_low),
            clip_sigma_high=float(clip_sigma_high),
        )
        clip_records.append(
            {
                "frame": int(frame_idx + 1),
                "sample_median": float(med),
                "sample_sigma_robust": float(sigma),
                "clip_lo": float(lo),
                "clip_hi": float(hi),
            }
        )

        counts_f = np.where(finite, counts, med)
        wins = np.clip(counts_f, lo, hi).astype(np.float32, copy=False)
        wins_sq = (wins * wins).astype(np.float32, copy=False)
        hot_mask = (counts_f > hi).astype(np.float32, copy=False)
        finite_mask = finite.astype(np.float32, copy=False)

        kernel_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for kernel in kernels_sorted:
            mu = box_mean(wins, int(kernel)).astype(np.float64, copy=False)
            m2 = box_mean(wins_sq, int(kernel)).astype(np.float64, copy=False)
            var = np.maximum(0.0, m2 - mu * mu)
            kk = int(kernel) * int(kernel)
            if kk > 1:
                var *= float(kk) / float(kk - 1)
            hot_frac = box_mean(hot_mask, int(kernel)).astype(np.float64, copy=False)
            finite_frac = box_mean(finite_mask, int(kernel)).astype(np.float64, copy=False)
            kernel_cache[int(kernel)] = (mu, var, hot_frac, finite_frac)

        for kernel, entries in by_kernel.items():
            mu_map, var_map, hot_frac_map, finite_frac_map = kernel_cache[int(kernel)]
            for state in entries:
                mu = mu_map[state.yy, state.xx]
                var = var_map[state.yy, state.xx]
                hot_frac = hot_frac_map[state.yy, state.xx]
                finite_frac = finite_frac_map[state.yy, state.xx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    gain_local = var / np.clip(mu, 1e-12, None)
                ok = (
                    np.isfinite(gain_local)
                    & np.isfinite(mu)
                    & np.isfinite(var)
                    & np.isfinite(hot_frac)
                    & np.isfinite(finite_frac)
                    & (mu >= float(min_local_mean))
                    & (var >= float(min_local_var))
                    & (finite_frac >= float(min_finite_frac))
                    & (hot_frac <= float(max_hot_frac))
                    & (gain_local >= float(gain_min))
                    & (gain_local <= float(gain_max))
                )
                mgain = int(np.count_nonzero(ok))
                state.frame_mgain.append(mgain)
                if mgain == 0:
                    state.frame_gain.append(float("nan"))
                else:
                    vals = gain_local[ok].astype(np.float64, copy=False)
                    weights = np.clip(mu[ok].astype(np.float64, copy=False), 1e-6, None)
                    gain_med = weighted_quantile(vals, weights, 0.5)
                    state.frame_gain.append(float(gain_med))
                    state.table_sum[ok] += vals
                    state.table_count[ok] += 1

        frame_numbers.append(int(frame_idx + 1))

    return states, frame_numbers, clip_records


def summarize_state(
    state: ConfigState,
    frame_numbers: Sequence[int],
    *,
    prior_gains: Optional[Sequence[float]],
) -> Dict[str, Any]:
    del frame_numbers  # kept for possible future expansion
    frame_arr = np.asarray(state.frame_gain, dtype=np.float64)
    mg_arr = np.asarray(state.frame_mgain, dtype=np.float64)
    valid_frame = np.isfinite(frame_arr) & (frame_arr > 0)
    frame_vals = frame_arr[valid_frame]

    table = np.full_like(state.table_sum, np.nan, dtype=np.float64)
    ok_table = state.table_count > 0
    if np.any(ok_table):
        table[ok_table] = state.table_sum[ok_table] / np.clip(state.table_count[ok_table], 1, None)
    table_vals = table[np.isfinite(table) & (table > 0)]

    table_stats = _quant_dict(np.asarray(table_vals, dtype=np.float64))
    frame_stats = _quant_dict(np.asarray(frame_vals, dtype=np.float64))
    mgain_stats = _quant_dict(mg_arr[np.isfinite(mg_arr)])

    valid_grid_fraction = float(np.count_nonzero(np.isfinite(table))) / float(max(1, state.gx * state.gy))
    valid_frame_fraction = float(np.count_nonzero(valid_frame)) / float(max(1, frame_arr.size))
    mgain_fraction = float(mgain_stats["median"]) / float(max(1, state.gx * state.gy)) if np.isfinite(mgain_stats["median"]) else 0.0

    table_med = table_stats["median"]
    frame_med = frame_stats["median"]
    table_spread = (
        float(max(0.0, table_stats["q90"] - table_stats["q10"]) / max(table_med, 1e-6))
        if np.isfinite(table_med)
        else float("inf")
    )
    frame_spread = (
        float(max(0.0, frame_stats["q90"] - frame_stats["q10"]) / max(frame_med, 1e-6))
        if np.isfinite(frame_med)
        else float("inf")
    )

    usable_priors = [
        float(value)
        for value in (prior_gains or [])
        if value is not None and np.isfinite(value) and float(value) > 0
    ]
    closest_prior_gain = None
    prior_penalty = 0.0
    if usable_priors and np.isfinite(table_med) and table_med > 0:
        closest_prior_gain = min(usable_priors, key=lambda value: abs(math.log(float(table_med) / float(value))))
        prior_penalty = abs(math.log(float(table_med) / float(closest_prior_gain))) / math.log(4.0)

    internal_score = (
        3.0 * valid_grid_fraction
        + 2.0 * valid_frame_fraction
        + 1.5 * mgain_fraction
        - 1.5 * min(table_spread, 5.0)
        - 1.0 * min(frame_spread, 5.0)
        - 0.25 * prior_penalty
    )
    consensus_weight = max(1e-6, (valid_grid_fraction + 0.05) * (valid_frame_fraction + 0.05) * (mgain_fraction + 0.05))
    consensus_weight /= (1.0 + max(0.0, table_spread)) * (1.0 + max(0.0, frame_spread))
    if prior_penalty > 0:
        consensus_weight *= math.exp(-0.20 * prior_penalty)

    return {
        "key": state.key,
        "kernel": int(state.kernel),
        "grid": [int(state.gx), int(state.gy)],
        "frame_gain_stats": frame_stats,
        "frame_mgain_stats": mgain_stats,
        "table_gain_stats": table_stats,
        "valid_grid_fraction": float(valid_grid_fraction),
        "valid_frame_fraction": float(valid_frame_fraction),
        "mgain_fraction": float(mgain_fraction),
        "table_gain_map": table.tolist(),
        "table_count_map": state.table_count.tolist(),
        "prior_gains": [float(value) for value in usable_priors],
        "closest_prior_gain": float(closest_prior_gain) if closest_prior_gain is not None else None,
        "prior_penalty": float(prior_penalty),
        "metadata_penalty": float(prior_penalty),
        "internal_score": float(internal_score),
        "consensus_weight": float(consensus_weight),
    }


def estimate_gain_from_binned_stack(
    stack: np.ndarray,
    *,
    metadata_gain: Optional[float] = None,
    prior_gains: Optional[Sequence[float]] = None,
    kernels: Sequence[int] = (11, 15, 21, 27, 35),
    grids: Sequence[Tuple[int, int]] = ((24, 24), (32, 32), (48, 48), (64, 64), (80, 80)),
    sample_pixels_for_clip: int = 200_000,
    clip_sigma_low: float = 6.0,
    clip_sigma_high: float = 6.0,
    min_local_mean: float = 0.25,
    min_local_var: float = 1e-6,
    min_finite_frac: float = 0.995,
    max_hot_frac: float = 0.06,
    gain_min: float = 0.01,
    gain_max: float = 1000.0,
    clip_output_min: float = 0.25,
    clip_output_max: float = 1000.0,
    seed: int = 0,
    ) -> Dict[str, Any]:
    effective_priors: List[float] = []
    for value in (prior_gains or []):
        if value is None or not np.isfinite(value) or float(value) <= 0:
            continue
        value_f = float(value)
        if any(math.isclose(value_f, existing, rel_tol=1e-6, abs_tol=1e-9) for existing in effective_priors):
            continue
        effective_priors.append(value_f)
    if metadata_gain is not None and np.isfinite(metadata_gain) and float(metadata_gain) > 0:
        metadata_gain_f = float(metadata_gain)
        if not any(math.isclose(metadata_gain_f, existing, rel_tol=1e-6, abs_tol=1e-9) for existing in effective_priors):
            effective_priors.append(metadata_gain_f)

    states, frame_numbers, clip_records = _estimate_from_array(
        stack=stack,
        kernels=kernels,
        grids=grids,
        sample_pixels_for_clip=sample_pixels_for_clip,
        clip_sigma_low=clip_sigma_low,
        clip_sigma_high=clip_sigma_high,
        min_local_mean=min_local_mean,
        min_local_var=min_local_var,
        min_finite_frac=min_finite_frac,
        max_hot_frac=max_hot_frac,
        gain_min=gain_min,
        gain_max=gain_max,
        seed=seed,
    )
    results = [summarize_state(state, frame_numbers, prior_gains=effective_priors) for state in states]
    valid_results = [
        result
        for result in results
        if np.isfinite(result["table_gain_stats"]["mean"]) and result["table_gain_stats"]["mean"] > 0
    ]
    if not valid_results:
        fallback = float(metadata_gain) if metadata_gain is not None and np.isfinite(metadata_gain) and metadata_gain > 0 else 1.0
        return {
            "recommended_gain": float(np.clip(fallback, clip_output_min, clip_output_max)),
            "selected_source": "metadata_fallback" if fallback != 1.0 else "default_fallback",
            "results": results,
            "run": {"frame_numbers": frame_numbers, "clip_records": clip_records, "frames_processed": len(frame_numbers)},
            "summary": {
                "n_valid_configs": 0,
                "prior_gains": effective_priors,
                "selected_gain": float(np.clip(fallback, clip_output_min, clip_output_max)),
                "selected_source": "metadata_fallback" if fallback != 1.0 else "default_fallback",
            },
        }

    candidate_values = np.asarray([float(result["table_gain_stats"]["mean"]) for result in valid_results], dtype=np.float64)
    candidate_weights = np.asarray([float(result["consensus_weight"]) for result in valid_results], dtype=np.float64)
    recommended = weighted_quantile(candidate_values, candidate_weights, 0.5)
    recommended = float(np.clip(recommended, clip_output_min, clip_output_max))
    nearest = min(valid_results, key=lambda result: abs(float(result["table_gain_stats"]["mean"]) - recommended))

    consensus_q10 = weighted_quantile(candidate_values, candidate_weights, 0.10)
    consensus_q90 = weighted_quantile(candidate_values, candidate_weights, 0.90)
    best_by_score = max(valid_results, key=lambda result: float(result["internal_score"]))

    return {
        "recommended_gain": float(recommended),
        "adaptive_candidate_gain": float(recommended),
        "selected_source": "adaptive_consensus",
        "best_result": best_by_score,
        "nearest_result": nearest,
        "results": results,
        "run": {
            "frame_numbers": frame_numbers,
            "clip_records": clip_records,
            "frames_processed": len(frame_numbers),
        },
        "summary": {
            "n_valid_configs": int(len(valid_results)),
            "candidate_gain_stats": {
                "mean": float(np.mean(candidate_values)),
                "median": float(np.median(candidate_values)),
                "q10": float(consensus_q10),
                "q90": float(consensus_q90),
                "min": float(np.min(candidate_values)),
                "max": float(np.max(candidate_values)),
            },
            "metadata_gain": float(metadata_gain) if metadata_gain is not None and np.isfinite(metadata_gain) else None,
            "prior_gains": effective_priors,
            "selected_gain": float(recommended),
            "selected_source": "adaptive_consensus",
        },
    }
