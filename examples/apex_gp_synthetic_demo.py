"""Synthetic demo of APEx-GP (no downloads).

Part 1 fits the GP core to fake pilot AUROCs drawn from a known power law plus
noise at sizes 60-360, then scores forecasts at 414-720 and 804-20000 with the
paper's metrics. Part 2 is a smoke test of the PyHealth ``APExGP`` wrapper on a
small synthetic ``SampleDataset``.

Run from the repository root:
    python examples/apex_gp_synthetic_demo.py
"""

from __future__ import annotations

import numpy as np
import torch

from pyhealth.models.apex_gp import (
    DEFAULT_SUBSET_SIZES,
    APExGP,
    APExGPRegressor,
)

TRUE_EPSILON, TRUE_THETA1, TRUE_THETA2 = 0.08, 0.9, -0.45
NOISE_SD = 0.01
TARGET_RANGES = {
    "414-720": np.array([414, 490, 566, 643, 720], dtype=float),
    "804-20000": np.geomspace(804, 20000, 6).round(),
}


def true_curve(sizes: np.ndarray) -> np.ndarray:
    """Ground-truth power-law AUROC curve.

    Args:
        sizes: Training-set sizes.

    Returns:
        Noise-free AUROC at each size.
    """
    return (1.0 - TRUE_EPSILON) - TRUE_THETA1 * sizes**TRUE_THETA2


def gp_demo(seed: int = 0) -> None:
    """Fits both mean functions to noisy pilot data and prints the metrics.

    Args:
        seed: Seed for the synthetic noise.
    """
    rng = np.random.default_rng(seed)
    pilot_sizes = np.asarray(DEFAULT_SUBSET_SIZES, dtype=float)
    pilot_aurocs = true_curve(pilot_sizes) + rng.normal(0, NOISE_SD, pilot_sizes.size)
    targets = {
        name: (sizes, true_curve(sizes) + rng.normal(0, NOISE_SD, sizes.size))
        for name, sizes in TARGET_RANGES.items()
    }
    print("Pilot sizes: ", pilot_sizes.astype(int).tolist())
    print("Pilot AUROCs:", np.round(pilot_aurocs, 4).tolist())

    for mean_function in ("power_law", "arctan"):
        gp = APExGPRegressor(mean_function=mean_function, seed=seed)
        gp.fit(pilot_sizes, pilot_aurocs)
        print(f"\n== mean_function={mean_function} ==")
        print("  fitted:", gp.hyperparameters_)
        for name, (sizes, observed) in targets.items():
            metrics = gp.evaluate(sizes, observed)
            mean, lower, upper = gp.predict_interval(sizes)
            print(
                f"  {name:>10}: RMSE={metrics['rmse']:.4f}  "
                f"quantized lik.={metrics['quantized_likelihood']:.3f}  "
                f"95% coverage={metrics['coverage']:.2f}"
            )
            print(
                f"{'':14}largest size {int(sizes[-1])}: true "
                f"{observed[-1]:.3f}, predicted {mean[-1]:.3f} "
                f"[{lower[-1]:.3f}, {upper[-1]:.3f}]"
            )


def wrapper_smoke_test(seed: int = 0) -> None:
    """Fits the PyHealth wrapper on synthetic band-power-like features.

    Args:
        seed: Seed for the synthetic data.
    """
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    rng = np.random.default_rng(seed)
    n_subjects, per_subject, n_features = 20, 60, 80
    samples = []
    for subject in range(n_subjects):
        offset = rng.normal(0, 0.5, n_features)  # subject-specific shift
        for w in range(per_subject):
            label = w % 2
            x = offset + rng.normal(0, 1.0, n_features)
            x[:8] += 0.4 * label  # weak class signal in a few features
            samples.append(
                {
                    "patient_id": f"S{subject:03d}",
                    "features": x.astype(np.float32).tolist(),
                    "label": label,
                }
            )
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"label": "binary"},
    )
    model = APExGP(dataset, n_seeds=2, n_restarts=2, n_iterations=500, seed=seed)
    model.fit()
    print("\n== APExGP wrapper smoke test ==")
    print("  pilot AUROCs:", np.round(model.pilot_results_.auroc_mean, 4).tolist())
    mean, lower, upper = model.predict_interval([720, 1440, 5000])
    for size, m, lo, hi in zip([720, 1440, 5000], mean, lower, upper, strict=True):
        print(f"  size {size:>5}: {m:.3f} [{lo:.3f}, {hi:.3f}]")

    batch = next(iter(get_dataloader(dataset, batch_size=32, shuffle=False)))
    with torch.no_grad():
        out = model(**batch)
    print(
        "  forward:",
        {k: tuple(v.shape) for k, v in out.items()},
        f"loss={out['loss'].item():.4f}",
    )


if __name__ == "__main__":
    gp_demo()
    wrapper_smoke_test()
