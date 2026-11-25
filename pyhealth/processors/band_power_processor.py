"""Log band-power features for multichannel signals (EEG, ECG, EMG, ...)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from scipy.integrate import trapezoid
from scipy.signal import welch

from . import register_processor
from .base_processor import LegacyFeatureProcessor

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha_mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
"""Classic EEG bands in Hz, as (low, high). Dict order defines feature order."""


def validate_bands(
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


def _to_numpy(value: Any) -> np.ndarray:
    """Converts a tensor, array or nested list to NumPy (float64 if floating).

    Args:
        value: A ``torch.Tensor``, ``np.ndarray`` or array-like object.

    Returns:
        The data as a ``np.ndarray``; floating-point inputs become float64.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64, copy=False)
    return array


def log_band_power(
    signal: Any,
    sample_rate: float,
    bands: Mapping[str, tuple[float, float]] = DEFAULT_BANDS,
    nperseg: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-channel log band power of one window via Welch's method.

    The PSD is integrated over each band [low, high] with the trapezoidal rule,
    then log-transformed. With the default ``nperseg`` (one second of data),
    frequency resolution is 1 Hz and a 2 s window yields 3 overlapping
    segments.

    Args:
        signal: Array-like of shape (n_channels, n_times); a tensor is fine.
        sample_rate: Sampling frequency in Hz.
        bands: Mapping from band name to (low, high) in Hz. Dict order defines
            feature order.
        nperseg: Welch segment length in samples. Defaults to
            ``min(n_times, round(sample_rate))``.
        eps: Small constant added before the log to avoid ``log(0)``.

    Returns:
        A float64 array of shape (n_channels * n_bands,), channel-major (all
        bands of channel 0, then channel 1, ...).

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
    validate_bands(bands, fs)

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


@register_processor("band_power")
class BandPowerProcessor(LegacyFeatureProcessor):
    """Feature processor turning a multichannel signal into log band powers.

    Input:
        - One window of shape (n_channels, n_times): a ``torch.Tensor``,
          ``np.ndarray`` or nested list of numbers, sampled at ``sample_rate``.

    Processing:
        - Welch PSD per channel (``scipy.signal.welch``), integrated over each
          band with the trapezoidal rule, then ``log(power + eps)``.
          Computation is in float64; the result is cast to ``dtype``.
        - ``fit()`` learns nothing from the data (it only records the channel
          count for ``size()``), so fitting on all samples leaks nothing.

    Output:
        - 1-D ``torch.Tensor`` of length n_channels * n_bands, channel-major
          (see ``feature_names``).

    Args:
        sample_rate: Sampling frequency of the input in Hz.
        bands: Mapping from band name to (low, high) in Hz. Defaults to
            ``DEFAULT_BANDS`` (delta, theta, alpha/mu, beta, gamma).
        nperseg: Welch segment length in samples. Defaults to
            ``min(n_times, round(sample_rate))`` (1 Hz resolution).
        eps: Constant added before the log.
        dtype: Output tensor dtype.

    Raises:
        ValueError: If a band is malformed or above the Nyquist frequency, or
            ``nperseg`` is smaller than 2.

    Examples:
        >>> import torch
        >>> from pyhealth.processors import BandPowerProcessor
        >>> processor = BandPowerProcessor(sample_rate=200.0)
        >>> processor.process(torch.randn(16, 400)).shape
        torch.Size([80])
    """

    stores_tensor = True

    def __init__(
        self,
        sample_rate: float = 200.0,
        bands: Mapping[str, tuple[float, float]] | None = None,
        nperseg: int | None = None,
        eps: float = 1e-12,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.bands: dict[str, tuple[float, float]] = dict(
            DEFAULT_BANDS if bands is None else bands
        )
        validate_bands(self.bands, self.sample_rate)
        if nperseg is not None and nperseg < 2:
            raise ValueError(f"nperseg must be >= 2; got {nperseg}.")
        self.nperseg = nperseg
        self.eps = eps
        self.dtype = dtype
        self._n_channels: int | None = None

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Records the channel count from the first sample with this field.

        Args:
            samples: Samples to inspect.
            field: Name of the signal field.
        """
        for sample in samples:
            value = sample.get(field)
            if value is not None:
                self._n_channels = int(_to_numpy(value).shape[0])
                return

    def process(self, value: Any) -> torch.Tensor:
        """Computes the log band-power vector of one window.

        Args:
            value: Signal of shape (n_channels, n_times).

        Returns:
            Tensor of shape (n_channels * n_bands,) with dtype ``self.dtype``.
        """
        features = log_band_power(
            value, self.sample_rate, self.bands, self.nperseg, self.eps
        )
        return torch.as_tensor(features, dtype=self.dtype)

    def feature_names(self, channel_names: Sequence[str]) -> list[str]:
        """Column names matching ``process()`` output.

        Args:
            channel_names: Channel names in signal row order.

        Returns:
            ``"{channel}_{band}"`` names, channel-major.
        """
        return [f"{ch}_{band}" for ch in channel_names for band in self.bands]

    def size(self) -> int | None:
        """Output length, or None before ``fit()`` has seen a sample."""
        if self._n_channels is None:
            return None
        return self._n_channels * len(self.bands)

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def __repr__(self) -> str:
        return (
            f"BandPowerProcessor(sample_rate={self.sample_rate}, "
            f"bands={list(self.bands)}, nperseg={self.nperseg})"
        )
