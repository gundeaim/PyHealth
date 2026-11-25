import unittest

import numpy as np
import torch
from scipy.integrate import trapezoid
from scipy.signal import welch

from pyhealth.processors import BandPowerProcessor, get_processor
from pyhealth.processors.band_power_processor import DEFAULT_BANDS

FS = 200.0
N_CHANNELS, N_TIMES = 16, 400  # 2 s windows, as in EEGMotorImageryEEGBCI


def _noise(seed: int = 0) -> np.ndarray:
    """Deterministic white noise of shape (16, 400)."""
    return np.random.default_rng(seed).standard_normal((N_CHANNELS, N_TIMES))


def _sine(freq: float, amplitude: float = 1.0, seed: int = 0) -> np.ndarray:
    """A sine at ``freq`` Hz on every channel plus weak deterministic noise."""
    t = np.arange(N_TIMES) / FS
    phases = np.linspace(0.0, np.pi, N_CHANNELS)[:, None]
    return amplitude * np.sin(2 * np.pi * freq * t + phases) + 0.05 * _noise(seed)


class TestBandPowerProcessor(unittest.TestCase):
    """Log band-power features from one (n_channels, n_times) window."""

    def setUp(self):
        self.processor = BandPowerProcessor(sample_rate=FS)

    def test_registered_under_band_power(self):
        self.assertIs(get_processor("band_power"), BandPowerProcessor)

    def test_output_shape_and_dtype(self):
        out = self.processor.process(torch.as_tensor(_noise(), dtype=torch.float32))
        self.assertEqual(tuple(out.shape), (N_CHANNELS * len(DEFAULT_BANDS),))
        self.assertEqual(out.dtype, torch.float32)
        out64 = BandPowerProcessor(sample_rate=FS, dtype=torch.float64).process(
            _noise()
        )
        self.assertEqual(out64.dtype, torch.float64)

    def test_ten_hz_sine_has_most_alpha_mu_power(self):
        out = np.asarray(self.processor.process(_sine(10.0)))
        per_channel = out.reshape(N_CHANNELS, len(DEFAULT_BANDS))
        alpha = list(DEFAULT_BANDS).index("alpha_mu")
        np.testing.assert_array_equal(per_channel.argmax(axis=1), alpha)

    def test_matches_direct_welch_trapezoid_computation(self):
        """Reference: Welch PSD (1 s segments) integrated with trapezoid, log."""
        x = _noise(seed=3)
        freqs, psd = welch(x, fs=FS, nperseg=int(FS), axis=-1)
        expected = []
        for channel in psd:
            for low, high in DEFAULT_BANDS.values():
                mask = (freqs >= low) & (freqs <= high)
                expected.append(np.log(trapezoid(channel[mask], freqs[mask]) + 1e-12))
        processor = BandPowerProcessor(sample_rate=FS, dtype=torch.float64)
        np.testing.assert_allclose(
            np.asarray(processor.process(x)), expected, rtol=0, atol=1e-12
        )

    def test_tensor_and_numpy_inputs_agree(self):
        x = _noise(seed=1).astype(np.float32)
        from_numpy = np.asarray(self.processor.process(x))
        from_tensor = np.asarray(self.processor.process(torch.as_tensor(x)))
        np.testing.assert_array_equal(from_numpy, from_tensor)

    def test_configurable_bands_change_length(self):
        mu_only = BandPowerProcessor(sample_rate=FS, bands={"mu": (8.0, 13.0)})
        self.assertEqual(tuple(mu_only.process(_noise()).shape), (N_CHANNELS,))
        three = BandPowerProcessor(
            sample_rate=FS,
            bands={"theta": (4.0, 8.0), "mu": (8.0, 13.0), "beta": (13.0, 30.0)},
        )
        self.assertEqual(tuple(three.process(_noise()).shape), (N_CHANNELS * 3,))

    def test_fit_sets_size_and_learns_no_statistics(self):
        self.assertIsNone(self.processor.size())
        self.processor.fit([{"signal": None}, {"signal": _noise()}], "signal")
        self.assertEqual(self.processor.size(), N_CHANNELS * len(DEFAULT_BANDS))
        self.assertFalse(self.processor.learns_statistics)
        self.assertTrue(self.processor.stores_tensor)

    def test_feature_names_are_channel_major(self):
        names = self.processor.feature_names(["C3", "C4"])
        self.assertEqual(names[:2], ["C3_delta", "C3_theta"])
        self.assertEqual(names[5], "C4_delta")
        self.assertEqual(len(names), 2 * len(DEFAULT_BANDS))

    def test_legacy_interface(self):
        self.assertFalse(self.processor.is_token())
        self.assertEqual(self.processor.schema(), ("value",))
        self.assertEqual(self.processor.dim(), (1,))
        self.assertEqual(self.processor.spatial(), (False,))

    def test_wrong_ndim_raises(self):
        for shape in ((N_TIMES,), (2, N_CHANNELS, N_TIMES)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.processor.process(np.zeros(shape))

    def test_non_finite_values_raise(self):
        for bad in (np.nan, np.inf):
            x = _noise()
            x[3, 10] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.processor.process(x)

    def test_invalid_bands_raise(self):
        cases = {
            "above Nyquist": ({"gamma": (30.0, 45.0)}, 80.0),  # Nyquist 40 Hz
            "low >= high": ({"bad": (13.0, 8.0)}, FS),
            "negative low": ({"bad": (-1.0, 4.0)}, FS),
            "empty": ({}, FS),
        }
        for name, (bands, rate) in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                BandPowerProcessor(sample_rate=rate, bands=bands)

    def test_band_with_too_few_psd_bins_raises(self):
        """nperseg=50 gives 4 Hz bins, so delta (1-4 Hz) holds a single bin."""
        processor = BandPowerProcessor(sample_rate=FS, nperseg=50)
        with self.assertRaises(ValueError):
            processor.process(_noise())

    def test_invalid_nperseg_raises(self):
        with self.assertRaises(ValueError):
            BandPowerProcessor(sample_rate=FS, nperseg=1)


if __name__ == "__main__":
    unittest.main()
