from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from weather_agent.models import HAZARDS, clamp01


def fit_histogram_calibrator(samples: Iterable[Tuple[float, int]], bins: int = 12) -> Dict[str, object]:
    pts = [(clamp01(float(p)), int(y)) for p, y in samples]
    n = len(pts)
    if n == 0:
        return {
            "method": "histogram",
            "bins": bins,
            "bin_rate": [0.5] * bins,
            "bin_count": [0] * bins,
            "global_rate": 0.5,
            "alpha": 0.0,
            "samples": 0,
            "ece": 0.0,
            "reliability": [],
        }

    global_rate = (sum(y for _, y in pts) + 1.0) / (n + 2.0)
    pos = [0.0] * bins
    cnt = [0.0] * bins
    p_sum = [0.0] * bins
    for p, y in pts:
        idx = min(bins - 1, int(p * bins))
        pos[idx] += y
        cnt[idx] += 1.0
        p_sum[idx] += p

    rates: List[float] = []
    reliability: List[dict] = []
    ece = 0.0
    for i in range(bins):
        rate = (pos[i] + 2.0 * global_rate) / (cnt[i] + 2.0)
        rate = clamp01(rate)
        rates.append(rate)
        if cnt[i] <= 0:
            continue
        mean_conf = p_sum[i] / cnt[i]
        weight = cnt[i] / n
        ece += weight * abs(mean_conf - rate)
        reliability.append(
            {
                "bin": i,
                "count": int(cnt[i]),
                "mean_confidence": round(mean_conf, 4),
                "event_rate": round(rate, 4),
            }
        )

    alpha = clamp01(min(0.85, n / 500.0))
    return {
        "method": "histogram",
        "bins": bins,
        "bin_rate": rates,
        "bin_count": [int(x) for x in cnt],
        "global_rate": round(global_rate, 6),
        "alpha": round(alpha, 6),
        "samples": n,
        "ece": round(ece, 6),
        "reliability": reliability,
    }


def fit_beta_calibrator(samples: Iterable[Tuple[float, int]]) -> Dict[str, object]:
    """
    Lightweight beta-like calibrator without external optimizer.
    It learns negative/positive prediction means and linearly rescales probabilities,
    then shrinks towards identity when sample size is small.
    """
    pts = [(clamp01(float(p)), int(y)) for p, y in samples]
    n = len(pts)
    if n == 0:
        return {
            "method": "beta",
            "neg_mean": 0.25,
            "pos_mean": 0.75,
            "alpha": 0.0,
            "samples": 0,
            "ece": 0.0,
            "reliability": [],
        }

    pos = [p for p, y in pts if y == 1]
    neg = [p for p, y in pts if y == 0]
    pos_mean = sum(pos) / len(pos) if pos else 0.75
    neg_mean = sum(neg) / len(neg) if neg else 0.25
    if pos_mean <= neg_mean:
        pos_mean = min(0.95, neg_mean + 0.1)

    alpha = clamp01(min(0.85, n / 500.0))

    # compute ECE on deciles for audit
    bins = 10
    pos_bin = [0] * bins
    cnt = [0] * bins
    p_sum = [0.0] * bins
    for p, y in pts:
        idx = min(bins - 1, int(p * bins))
        pos_bin[idx] += int(y)
        cnt[idx] += 1
        p_sum[idx] += p

    ece = 0.0
    reliability = []
    for i in range(bins):
        if cnt[i] == 0:
            continue
        mean_conf = p_sum[i] / cnt[i]
        event_rate = pos_bin[i] / cnt[i]
        w = cnt[i] / n
        ece += w * abs(mean_conf - event_rate)
        reliability.append(
            {
                "bin": i,
                "count": cnt[i],
                "mean_confidence": round(mean_conf, 4),
                "event_rate": round(event_rate, 4),
            }
        )

    return {
        "method": "beta",
        "neg_mean": round(neg_mean, 6),
        "pos_mean": round(pos_mean, 6),
        "alpha": round(alpha, 6),
        "samples": n,
        "ece": round(ece, 6),
        "reliability": reliability,
    }


def apply_histogram_calibrator(prob: float, calibrator: Dict[str, object] | None) -> float:
    p = clamp01(float(prob))
    if not calibrator:
        return p

    bins = int(calibrator.get("bins", 0) or 0)
    rates = calibrator.get("bin_rate")
    alpha = clamp01(float(calibrator.get("alpha", 0.0)))
    if bins <= 0 or not isinstance(rates, list) or not rates:
        return p

    idx = min(bins - 1, int(p * bins))
    empirical = clamp01(float(rates[idx]))
    return clamp01((1.0 - alpha) * p + alpha * empirical)


def apply_beta_calibrator(prob: float, calibrator: Dict[str, object] | None) -> float:
    p = clamp01(float(prob))
    if not calibrator:
        return p

    neg_mean = float(calibrator.get("neg_mean", 0.25))
    pos_mean = float(calibrator.get("pos_mean", 0.75))
    alpha = clamp01(float(calibrator.get("alpha", 0.0)))

    denom = max(1e-6, pos_mean - neg_mean)
    mapped = clamp01((p - neg_mean) / denom)
    return clamp01((1.0 - alpha) * p + alpha * mapped)


def calibrate_hazard_probs(raw_probs: Dict[str, float], calibrators: Dict[str, Dict[str, object]] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for h in HAZARDS:
        p = float(raw_probs.get(h, 0.0))
        c = calibrators.get(h) if calibrators else None
        method = str((c or {}).get("method", "histogram")).strip().lower()
        if method == "beta":
            out[h] = apply_beta_calibrator(p, c)
        else:
            out[h] = apply_histogram_calibrator(p, c)
    return out
