"""Band-power features, subject splits and nested pilot subsets for EEGBCI.

Turns the windowed samples produced by PyHealth's ``EEGMotorImageryEEGBCI`` task
into a feature matrix suitable for training a logistic regression classifier on
nested subsets of increasing size, as required by an APEx-GP style learning-curve
extrapolation experiment (Harvey et al., 2023).

Pipeline:
    1. ``filter_motor_imagery`` keeps only windows from imagery runs.
    2. ``event_code_to_binary`` maps T0 -> 0 (rest), T1/T2 -> 1 (imagery).
    3. ``compute_log_band_power`` turns one (n_channels, n_times) window into
       n_channels * n_bands log band-power features (Welch PSD).
    4. ``build_feature_matrix`` stacks everything into ``X, y, groups``.
    5. ``split_by_subject`` makes subject-disjoint train/val/test index sets.
    6. ``draw_nested_subsets`` draws nested training subsets (counted in
       windows), each containing both classes.

Run ``python eegbci_features.py`` to execute the synthetic self-check.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

try:  # torch is only needed to recognise tensors; it is not otherwise required.
    import torch
except ImportError:  # pragma: no cover - exercised only when torch is absent
    torch = None  # type: ignore[assignment]

MOTOR_IMAGERY_RUNS: frozenset[int] = frozenset({4, 6, 8, 10, 12, 14})
"""EEGBCI runs in which the subject imagines (rather than executes) movement."""

EVENT_CODE_TO_BINARY: dict[str, int] = {"T0": 0, "T1": 1, "T2": 1}
"""Binary label from the EDF annotation code: rest = 0, imagined movement = 1."""

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha_mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
"""Frequency bands in Hz, as (low, high). Order defines feature order."""

DEFAULT_SUBSET_SIZES: tuple[int, ...] = (60, 94, 147, 230, 360)
"""Pilot training-set sizes, in windows, mirroring the APEx-GP paper."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_numpy(value: Any) -> np.ndarray:
    """Converts a torch tensor, array or nested sequence to a NumPy array.

    Args:
        value: A ``torch.Tensor``, ``np.ndarray`` or array-like object.

    Returns:
        The data as a ``np.ndarray`` (float64 for floating-point inputs).
    """
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64, copy=False)
    return array


def _to_scalar(value: Any) -> Any:
    """Unwraps 0-d tensors/arrays to Python scalars; returns others unchanged.

    Args:
        value: A scalar, 0-d tensor or 0-d array.

    Returns:
        A plain Python scalar when ``value`` is 0-d, otherwise ``value``.
    """
    if torch is not None and isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_bands(
    bands: Mapping[str, tuple[float, float]], sample_rate: float
) -> None:
    """Checks that every band is well formed and below the Nyquist frequency.

    Args:
        bands: Mapping from band name to (low, high) in Hz.
        sample_rate: Sampling frequency in Hz.

    Raises:
        ValueError: If ``bands`` is empty, a band has low >= high, low < 0, or
            high exceeds the Nyquist frequency.
    """
    if not bands:
        raise ValueError("At least one frequency band is required.")
    nyquist = sample_rate / 2.0
    for name, (low, high) in bands.items():
        if not 0.0 <= low < high:
            raise ValueError(f"Band {name!r} must satisfy 0 <= low < high.")
        if high > nyquist:
            raise ValueError(
                f"Band {name!r} upper edge {high} Hz exceeds Nyquist "
                f"({nyquist} Hz) for sample_rate={sample_rate}."
            )


# --------------------------------------------------------------------------- #
# Labels and filtering
# --------------------------------------------------------------------------- #
def event_code_to_binary(event_code: str) -> int:
    """Maps an EEGBCI annotation code to the binary rest/imagery label.

    Args:
        event_code: The sample's ``event_code`` ("T0", "T1" or "T2").

    Returns:
        0 for rest ("T0"), 1 for imagined movement ("T1" or "T2").

    Raises:
        ValueError: If ``event_code`` is not one of the known codes.
    """
    code = str(_to_scalar(event_code)).strip().upper()
    try:
        return EVENT_CODE_TO_BINARY[code]
    except KeyError as err:
        raise ValueError(
            f"Unknown event_code {event_code!r}; expected one of "
            f"{sorted(EVENT_CODE_TO_BINARY)}."
        ) from err


def filter_motor_imagery(
    samples: Iterable[Mapping[str, Any]],
    runs: Iterable[int] = MOTOR_IMAGERY_RUNS,
) -> list[Mapping[str, Any]]:
    """Keeps only windows recorded during motor-imagery runs.

    Filtering uses the integer ``run`` field, not ``label_family`` (which is
    "motor_imagery" for rest windows too) or ``run_type``.

    Args:
        samples: Iterable of PyHealth sample dicts (e.g. a ``SampleDataset``).
        runs: Run numbers to keep. Defaults to the six imagery runs.

    Returns:
        A list of the samples whose ``run`` is in ``runs``.
    """
    keep = {int(r) for r in runs}
    return [s for s in samples if int(_to_scalar(s["run"])) in keep]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def feature_names(
    channel_names: Sequence[str],
    bands: Mapping[str, tuple[float, float]] = DEFAULT_BANDS,
) -> list[str]:
    """Returns column names matching the output of ``compute_log_band_power``.

    Features are channel-major: all bands of channel 0, then channel 1, etc.

    Args:
        channel_names: EEG channel names in signal row order.
        bands: Mapping from band name to (low, high) in Hz.

    Returns:
        A list of ``"{channel}_{band}"`` strings of length
        ``len(channel_names) * len(bands)``.
    """
    return [f"{ch}_{band}" for ch in channel_names for band in bands]


def compute_log_band_power(
    signal: Any,
    sample_rate: float,
    bands: Mapping[str, tuple[float, float]] = DEFAULT_BANDS,
    nperseg: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Computes per-channel log band power of one EEG window via Welch's method.

    The PSD is integrated over each band [low, high] with the trapezoidal rule,
    then log-transformed. With the default ``nperseg`` (one second of data),
    frequency resolution is 1 Hz and a 2 s window yields 3 overlapping
    segments.

    Args:
        signal: Array-like of shape (n_channels, n_times); a tensor is fine.
        sample_rate: Sampling frequency in Hz (use the sample's own rate).
        bands: Mapping from band name to (low, high) in Hz. Dict order defines
            feature order.
        nperseg: Welch segment length in samples. Defaults to
            ``min(n_times, round(sample_rate))``.
        eps: Small constant added before the log to avoid ``log(0)``.

    Returns:
        A float64 array of shape (n_channels * n_bands,), channel-major.

    Raises:
        ValueError: If ``signal`` is not 2-D, contains non-finite values, the
            bands are invalid, or a band contains fewer than two PSD bins.
    """
    x = _to_numpy(signal)
    if x.ndim != 2:
        raise ValueError(f"signal must be 2-D (channels, times); got {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains NaN or infinite values.")
    fs = float(sample_rate)
    _validate_bands(bands, fs)

    n_times = x.shape[1]
    seg = nperseg if nperseg is not None else min(n_times, int(round(fs)))
    freqs, psd = welch(x, fs=fs, nperseg=seg, axis=-1)  # psd: (n_ch, n_freqs)

    powers = np.empty((x.shape[0], len(bands)), dtype=np.float64)
    for j, (name, (low, high)) in enumerate(bands.items()):
        mask = (freqs >= low) & (freqs <= high)
        if mask.sum() < 2:
            raise ValueError(
                f"Band {name!r} ({low}-{high} Hz) has fewer than two PSD bins "
                f"at resolution {freqs[1] - freqs[0]:.3g} Hz; increase nperseg."
            )
        powers[:, j] = trapezoid(psd[:, mask], freqs[mask], axis=-1)
    return np.log(powers + eps).reshape(-1)


def build_feature_matrix(
    samples: Iterable[Mapping[str, Any]],
    bands: Mapping[str, tuple[float, float]] = DEFAULT_BANDS,
    runs: Iterable[int] = MOTOR_IMAGERY_RUNS,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds the feature matrix, binary labels and subject groups.

    Args:
        samples: Iterable of sample dicts from ``EEGMotorImageryEEGBCI``.
        bands: Mapping from band name to (low, high) in Hz.
        runs: Runs to keep. Defaults to the six motor-imagery runs.
        nperseg: Welch segment length; see ``compute_log_band_power``.

    Returns:
        A tuple ``(X, y, groups)`` where ``X`` has shape
        (n_windows, n_channels * n_bands), ``y`` is an int64 array of 0/1
        labels and ``groups`` holds each window's ``subject_id``.

    Raises:
        ValueError: If no samples remain after filtering, or windows disagree
            on channel count or channel names.
    """
    kept = filter_motor_imagery(samples, runs)
    if not kept:
        raise ValueError("No samples left after filtering to imagery runs.")

    ref_channels = list(kept[0].get("channel_names", []))
    rows, labels, groups = [], [], []
    for sample in kept:
        channels = list(sample.get("channel_names", []))
        if ref_channels and channels and channels != ref_channels:
            raise ValueError(
                f"Channel mismatch in {sample.get('trial_id')}: {channels} "
                f"!= {ref_channels}."
            )
        features = compute_log_band_power(
            sample["signal"],
            sample_rate=float(_to_scalar(sample["sample_rate"])),
            bands=bands,
            nperseg=nperseg,
        )
        if rows and features.shape != rows[0].shape:
            raise ValueError(
                f"Feature length mismatch in {sample.get('trial_id')}: "
                f"{features.shape} != {rows[0].shape}."
            )
        rows.append(features)
        labels.append(event_code_to_binary(sample["event_code"]))
        groups.append(_to_scalar(sample["subject_id"]))

    return (
        np.vstack(rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
    )


# --------------------------------------------------------------------------- #
# Splits and nested subsets
# --------------------------------------------------------------------------- #
def split_by_subject(
    groups: Sequence[Any] | np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Splits window indices into subject-disjoint train/val/test sets.

    Fractions apply to the number of subjects, not windows. Subjects are
    shuffled with ``seed``; validation and test sizes are rounded and the
    remainder goes to training. Each split gets at least one subject.

    Args:
        groups: Subject identifier for every window (from
            ``build_feature_matrix``).
        train_frac: Fraction of subjects for training.
        val_frac: Fraction of subjects for validation.
        test_frac: Fraction of subjects for testing.
        seed: Seed for the subject shuffle.

    Returns:
        A dict with keys "train", "val" and "test" mapping to sorted int64
        arrays of window indices.

    Raises:
        ValueError: If the fractions are negative or do not sum to 1, or there
            are fewer than three subjects.
    """
    fracs = (train_frac, val_frac, test_frac)
    if min(fracs) < 0 or not np.isclose(sum(fracs), 1.0):
        raise ValueError("Split fractions must be non-negative and sum to 1.")
    groups_arr = np.asarray(groups)
    subjects = np.unique(groups_arr)  # sorted, so the shuffle is reproducible
    n_subjects = len(subjects)
    if n_subjects < 3:
        raise ValueError(f"Need at least 3 subjects to split; got {n_subjects}.")

    n_test = max(1, int(round(test_frac * n_subjects)))
    n_val = max(1, int(round(val_frac * n_subjects)))
    if n_subjects - n_test - n_val < 1:
        raise ValueError("Too few subjects left for training with these fracs.")

    shuffled = np.random.default_rng(seed).permutation(subjects)
    subject_sets = {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }
    return {
        name: np.flatnonzero(np.isin(groups_arr, subs)).astype(np.int64)
        for name, subs in subject_sets.items()
    }


def _stratified_order(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Orders positions so every prefix roughly preserves class proportions.

    Args:
        labels: Class label for each position.
        rng: NumPy random generator.

    Returns:
        A permutation of ``range(len(labels))``.
    """
    keys = np.empty(len(labels), dtype=np.float64)
    for cls in np.unique(labels):
        pos = rng.permutation(np.flatnonzero(labels == cls))
        keys[pos] = (np.arange(len(pos)) + rng.random(len(pos))) / len(pos)
    return np.argsort(keys, kind="stable")


def draw_nested_subsets(
    y: np.ndarray,
    train_indices: np.ndarray,
    sizes: Sequence[int] = DEFAULT_SUBSET_SIZES,
    seed: int = 0,
    stratify: bool = False,
    max_attempts: int = 1000,
) -> list[np.ndarray]:
    """Draws nested training subsets of increasing size, counted in windows.

    One random ordering of the training windows is drawn, and subset ``k`` is
    its first ``sizes[k]`` elements, so each subset contains all smaller ones.
    With ``stratify=False`` the ordering is uniform, redrawn (rejection
    sampling) until the smallest subset contains both classes. With
    ``stratify=True`` every prefix approximately keeps the training class
    ratio.

    Args:
        y: Binary labels for all windows (indexed by ``train_indices``).
        train_indices: Indices of the training windows.
        sizes: Strictly increasing subset sizes in windows.
        seed: Seed for the ordering.
        stratify: Whether to keep class proportions in every prefix.
        max_attempts: Maximum redraws when ``stratify=False``.

    Returns:
        A list of int64 index arrays (into ``y``), one per size, in the order
        the windows were drawn.

    Raises:
        ValueError: If sizes are not strictly increasing, are smaller than 2,
            exceed the training set, the training set lacks a class, or no
            valid ordering is found within ``max_attempts``.
    """
    sizes = [int(s) for s in sizes]
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if not sizes or any(b <= a for a, b in zip(sizes, sizes[1:], strict=False)):
        raise ValueError(f"sizes must be strictly increasing; got {sizes}.")
    if sizes[0] < 2:
        raise ValueError("The smallest subset needs at least 2 windows.")
    if sizes[-1] > len(train_indices):
        raise ValueError(
            f"Largest subset ({sizes[-1]}) exceeds training windows "
            f"({len(train_indices)})."
        )
    train_labels = np.asarray(y)[train_indices]
    n_classes = len(np.unique(train_labels))
    if n_classes < 2:
        raise ValueError("Training split contains only one class.")

    rng = np.random.default_rng(seed)
    for _ in range(1 if stratify else max_attempts):
        if stratify:
            order = _stratified_order(train_labels, rng)
        else:
            order = rng.permutation(len(train_indices))
        if len(np.unique(train_labels[order[: sizes[0]]])) == n_classes:
            ordered = train_indices[order]
            return [ordered[:size].copy() for size in sizes]
    raise ValueError(
        "Could not draw a smallest subset containing both classes; the "
        "training set may be extremely imbalanced. Try stratify=True."
    )


# --------------------------------------------------------------------------- #
# Synthetic self-check
# --------------------------------------------------------------------------- #
CHANNEL_NAMES: list[str] = [
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3",
    "C1", "C2", "C4", "C6", "CP5", "CP3", "CP4", "CP6",
]  # fmt: skip

_EEGBCI_LABEL = {  # (run, event_code) -> eegbci_label, for realistic dicts
    **{(r, "T0"): 0 for r in range(1, 15)},
    **{(r, "T1"): 3 for r in (4, 8, 12)},
    **{(r, "T2"): 4 for r in (4, 8, 12)},
    **{(r, "T1"): 7 for r in (6, 10, 14)},
    **{(r, "T2"): 8 for r in (6, 10, 14)},
    **{(r, "T1"): 1 for r in (3, 7, 11)},
    **{(r, "T2"): 2 for r in (3, 7, 11)},
}


def make_synthetic_samples(
    n_subjects: int = 10,
    runs: Sequence[int] = (3, 4, 6),
    windows_per_run: int = 60,
    sample_rate: float = 200.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Creates fake EEGBCI sample dicts with the same keys and shapes.

    Rest windows carry a stronger 10 Hz (mu) rhythm than imagery windows,
    mimicking event-related desynchronisation, so a classifier can learn.
    Run 3 is motor execution and should be dropped by the filter.

    Args:
        n_subjects: Number of synthetic subjects.
        runs: Run numbers to generate.
        windows_per_run: Windows per run (half T0, a quarter each T1/T2).
        sample_rate: Sampling rate in Hz.
        seed: Seed for the random signals.

    Returns:
        A list of sample dicts; tensors are torch tensors when torch is
        installed, NumPy arrays otherwise.
    """
    rng = np.random.default_rng(seed)
    n_times = int(2.0 * sample_rate)
    t = np.arange(n_times) / sample_rate
    wrap = (lambda a: torch.as_tensor(a)) if torch is not None else np.asarray
    codes = ["T0", "T1", "T0", "T2"]
    samples = []
    for subj in range(1, n_subjects + 1):
        for run in runs:
            for w in range(windows_per_run):
                code = codes[w % 4]
                mu_amp = 1.0 if code == "T0" else 0.4
                phase = rng.uniform(0, 2 * np.pi, size=(16, 1))
                sig = rng.standard_normal((16, n_times)) * 0.5
                sig += mu_amp * np.sin(2 * np.pi * 10.0 * t + phase)
                samples.append({
                    "patient_id": f"S{subj:03d}",
                    "record_id": f"R{run:02d}",
                    "subject_id": subj,
                    "run": run,
                    "run_type": "synthetic",
                    "signal_file": f"S{subj:03d}R{run:02d}.edf",
                    "trial_id": f"S{subj:03d}_R{run:02d}_{w:04d}",
                    "event_code": code,
                    "task_label": "rest" if code == "T0" else "imagery",
                    "label_family": "motor_imagery",
                    "label": wrap(np.array(2 if code == "T0" else 0)),
                    "eegbci_label": _EEGBCI_LABEL[(run, code)],
                    "signal": wrap(sig.astype(np.float32)),
                    "channel_names": list(CHANNEL_NAMES),
                    "start_time": 2.0 * w,
                    "end_time": 2.0 * w + 2.0,
                    "sample_rate": sample_rate,
                    "stft": wrap(np.zeros((16, 100, 3), dtype=np.float32)),
                })  # fmt: skip
    return samples


def run_self_check() -> None:
    """Runs assertions on synthetic data covering the whole pipeline.

    Raises:
        AssertionError: If any check fails.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    samples = make_synthetic_samples()
    n_imagery = sum(s["run"] in MOTOR_IMAGERY_RUNS for s in samples)

    # Labels: mapping and error on unknown codes.
    assert [event_code_to_binary(c) for c in ("T0", "T1", "T2")] == [0, 1, 1]
    try:
        event_code_to_binary("T3")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown event code should raise ValueError.")

    # Features: run 3 dropped, shape 16 x 5 = 80, labels follow event_code.
    X, y, groups = build_feature_matrix(samples)
    kept = filter_motor_imagery(samples)
    assert X.shape == (n_imagery, 80), X.shape
    assert y.shape == groups.shape == (n_imagery,)
    assert np.all(np.isfinite(X))
    expected = np.array([0 if s["event_code"] == "T0" else 1 for s in kept])
    assert np.array_equal(y, expected)
    assert len(feature_names(CHANNEL_NAMES)) == X.shape[1]

    # Physiology sanity: rest windows have more mu power (column C3_alpha_mu).
    mu_col = feature_names(CHANNEL_NAMES).index("C3_alpha_mu")
    assert X[y == 0, mu_col].mean() > X[y == 1, mu_col].mean()

    # Configurable bands.
    X2, _, _ = build_feature_matrix(samples, bands={"mu": (8.0, 13.0)})
    assert X2.shape == (n_imagery, 16)

    # Subject-disjoint splits that cover every window exactly once.
    splits = split_by_subject(groups, seed=42)
    subj = {k: set(groups[v].tolist()) for k, v in splits.items()}
    assert not (subj["train"] & subj["val"])
    assert not (subj["train"] & subj["test"])
    assert not (subj["val"] & subj["test"])
    all_idx = np.concatenate(list(splits.values()))
    assert np.array_equal(np.sort(all_idx), np.arange(len(y)))
    assert np.array_equal(splits["train"], split_by_subject(groups, seed=42)["train"])

    # Nested subsets, both classes everywhere, reproducible, for both modes.
    for stratify in (False, True):
        subsets = draw_nested_subsets(y, splits["train"], seed=7, stratify=stratify)
        assert [len(s) for s in subsets] == list(DEFAULT_SUBSET_SIZES)
        for small, big in zip(subsets, subsets[1:], strict=False):
            assert set(small.tolist()) <= set(big.tolist())
        for subset in subsets:
            assert set(y[subset].tolist()) == {0, 1}
            assert set(subset.tolist()) <= set(splits["train"].tolist())
        again = draw_nested_subsets(y, splits["train"], seed=7, stratify=stratify)
        assert all(np.array_equal(a, b) for a, b in zip(subsets, again, strict=True))

    # End to end: logistic regression on the largest subset beats chance.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X[subsets[-1]], y[subsets[-1]])
    auroc = roc_auc_score(
        y[splits["test"]], model.predict_proba(X[splits["test"]])[:, 1]
    )
    assert auroc > 0.9, auroc

    print(
        f"Self-check passed: X={X.shape}, train/val/test subjects="
        f"{len(subj['train'])}/{len(subj['val'])}/{len(subj['test'])}, "
        f"subset sizes={[len(s) for s in subsets]}, test AUROC={auroc:.3f}, "
        f"torch={'yes' if torch is not None else 'no (NumPy arrays used)'}"
    )


if __name__ == "__main__":
    run_self_check()
