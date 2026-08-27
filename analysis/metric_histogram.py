from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from analysis.metric_core import metric_quantiles, valid_metric_samples


@dataclass(frozen=True)
class HistogramSpec:
    edges: np.ndarray
    scale: str
    lower: float
    upper: float


def sample_values(values, valid, max_samples=0, seed=0):
    """Return finite valid values with reproducible random sampling without replacement."""
    samples = valid_metric_samples(values, valid)
    return subsample_values(samples, max_samples=max_samples, seed=seed)


def subsample_values(samples, max_samples=0, seed=0):
    """Randomly cap an already collected finite distribution."""
    samples = torch.as_tensor(samples).reshape(-1)
    samples = samples[torch.isfinite(samples)]
    if samples.numel() == 0:
        return samples.detach().cpu()
    if max_samples <= 0 or samples.numel() <= max_samples:
        return samples.detach().cpu()

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(samples.numel(), size=max_samples, replace=False)
    indices = torch.as_tensor(sampled_indices, device=samples.device)
    return samples[indices].detach().cpu()


def build_histogram_spec(
    reference_samples,
    bins=160,
    percentile_min=0.1,
    percentile_max=99.9,
):
    """Create fixed linear histogram edges from a reference distribution."""
    if bins <= 0:
        raise ValueError("Histogram bin count must be positive")
    if not 0 <= percentile_min < percentile_max <= 100:
        raise ValueError("Expected 0 <= histogram percentile min < max <= 100")

    samples = torch.as_tensor(reference_samples).reshape(-1)
    samples = samples[torch.isfinite(samples)]
    if samples.numel() == 0:
        raise ValueError("Cannot build a histogram from an empty distribution")

    q = torch.tensor(
        [percentile_min / 100.0, percentile_max / 100.0],
        dtype=samples.dtype,
    )
    lower, upper = metric_quantiles(samples, q)
    lower = float(lower)
    upper = float(upper)

    if upper <= lower:
        lower = float(samples.min().item())
        upper = float(samples.max().item())
    if upper <= lower:
        padding = max(abs(lower) * 1e-6, torch.finfo(samples.dtype).eps)
        lower -= padding
        upper += padding

    edges = np.linspace(lower, upper, bins + 1, dtype=np.float64)

    return HistogramSpec(
        edges=edges,
        scale="linear",
        lower=lower,
        upper=upper,
    )


def histogram_counts(samples, spec):
    values = torch.as_tensor(samples).reshape(-1)
    values = values[torch.isfinite(values)].detach().cpu().numpy()
    underflow = int(np.count_nonzero(values < spec.lower))
    overflow = int(np.count_nonzero(values > spec.upper))
    in_range = values[(values >= spec.lower) & (values <= spec.upper)]
    counts, _ = np.histogram(in_range, bins=spec.edges)
    return counts.astype(np.int64), underflow, overflow, int(values.size)


def _histogram_payload(metric, spec, series, metadata=None):
    payload = {
        "metric": metric,
        "scale": spec.scale,
        "bin_edges": spec.edges.tolist(),
        "range_min": spec.lower,
        "range_max": spec.upper,
        "series": series,
    }
    if metadata:
        payload.update(metadata)
    return payload


def save_single_histogram(
    image_path,
    data_path,
    metric,
    samples,
    spec,
    title=None,
    x_label=None,
    metadata=None,
):
    """Save one metric distribution as PNG + JSON."""
    counts, underflow, overflow, sample_count = histogram_counts(samples, spec)
    series = {
        "values": {
            "counts": counts.tolist(),
            "sample_count": sample_count,
            "underflow_count": underflow,
            "overflow_count": overflow,
        }
    }
    payload = _histogram_payload(metric, spec, series, metadata)
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w") as stream:
        json.dump(payload, stream, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Histogram image skipped because matplotlib is unavailable: {exc}")
        return False

    centers = 0.5 * (spec.edges[:-1] + spec.edges[1:])
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.step(centers, counts, where="mid", linewidth=1.4)
    axis.set_xlabel(x_label or metric)
    axis.set_ylabel("Sample count")
    axis.set_title(title or f"Metric distribution: {metric}")
    axis.grid(alpha=0.2)
    note = []
    if underflow:
        note.append(f"{underflow:,} below range")
    if overflow:
        note.append(f"{overflow:,} above range")
    if note:
        axis.text(
            0.98,
            0.95,
            ", ".join(note),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    return True


def save_comparison_histogram(
    image_path,
    data_path,
    metric,
    reference_samples,
    target_samples,
    spec,
    title=None,
    x_label=None,
    reference_label="reference",
    target_label="target",
    metadata=None,
):
    """Save reference/target distributions using exactly the same bin edges."""
    ref_counts, ref_under, ref_over, ref_count = histogram_counts(
        reference_samples, spec
    )
    target_counts, target_under, target_over, target_count = histogram_counts(
        target_samples, spec
    )
    ref_fraction = (
        ref_counts.astype(np.float64) / ref_count
        if ref_count > 0
        else np.zeros_like(ref_counts, dtype=np.float64)
    )
    target_fraction = (
        target_counts.astype(np.float64) / target_count
        if target_count > 0
        else np.zeros_like(target_counts, dtype=np.float64)
    )
    series = {
        reference_label: {
            "counts": ref_counts.tolist(),
            "fractions": ref_fraction.tolist(),
            "sample_count": ref_count,
            "underflow_count": ref_under,
            "overflow_count": ref_over,
        },
        target_label: {
            "counts": target_counts.tolist(),
            "fractions": target_fraction.tolist(),
            "sample_count": target_count,
            "underflow_count": target_under,
            "overflow_count": target_over,
        },
    }
    payload = _histogram_payload(metric, spec, series, metadata)
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w") as stream:
        json.dump(payload, stream, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Histogram image skipped because matplotlib is unavailable: {exc}")
        return False

    centers = 0.5 * (spec.edges[:-1] + spec.edges[1:])
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.step(centers, ref_fraction, where="mid", linewidth=1.4, label=reference_label)
    axis.step(centers, target_fraction, where="mid", linewidth=1.4, label=target_label)
    axis.set_xlabel(x_label or metric)
    axis.set_ylabel("Fraction of sampled observations")
    axis.set_title(title or f"Metric distribution comparison: {metric}")
    axis.grid(alpha=0.2)
    axis.legend()

    notes = []
    if ref_under or ref_over:
        notes.append(f"{reference_label}: {ref_under:,} below / {ref_over:,} above")
    if target_under or target_over:
        notes.append(f"{target_label}: {target_under:,} below / {target_over:,} above")
    if notes:
        axis.text(
            0.98,
            0.95,
            "\n".join(notes),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    return True
