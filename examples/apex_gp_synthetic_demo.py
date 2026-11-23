"""Synthetic demo of APEx-GP (no downloads).

Part 1 fits the GP core to fake pilot AUROCs drawn from a known power law plus
noise at sizes 60-360, then scores forecasts at 414-720 and 804-20000 with the
paper's metrics, averaged over 20 synthetic repeats. With the authors' default
optimizer (Adam, lr=0.01) each fit runs up to ~20k iterations, so Part 1 takes
several minutes. Part 2 is a smoke test of the PyHealth ``APExGP`` wrapper on a
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


def gp_demo(n_repeats: int = 20, seed: int = 0) -> None:
    """Fits both mean functions on repeated synthetic draws and averages metrics.

    Each repeat draws fresh noisy pilot AUROCs at sizes 60-360 and fresh noisy
    "observed" AUROCs at the target sizes. Coverage from a single draw of 5-6
    points is too noisy to judge calibration, so metrics are averaged.

    Args:
        n_repeats: Number of synthetic repeats.
        seed: Base seed; repeat ``r`` uses ``seed + r``.
    """
    pilot_sizes = np.asarray(DEFAULT_SUBSET_SIZES, dtype=float)
    results: dict[tuple[str, str], list[dict[str, float]]] = {}
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        noise = rng.normal(0, NOISE_SD, pilot_sizes.size)
        pilot_aurocs = true_curve(pilot_sizes) + noise
        targets = {
            name: (sizes, true_curve(sizes) + rng.normal(0, NOISE_SD, sizes.size))
            for name, sizes in TARGET_RANGES.items()
        }
        for mean_function in ("power_law", "arctan"):
            gp = APExGPRegressor(mean_function=mean_function, seed=seed + rep)
            gp.fit(pilot_sizes, pilot_aurocs)
            if rep == 0:
                hp = gp.hyperparameters_
                print(
                    f"[repeat 0, {mean_function}] sigma prior (m, s) = "
                    f"({gp.sigma_prior_loc_:.4f}, {gp.sigma_prior_scale_:.4f}); "
                    f"fitted sigma={hp.sigma:.4f}, lengthscale={hp.lengthscale:.4g}, "
                    f"tau={hp.tau:.4f}; iterations per restart="
                    f"{gp.n_iterations_run_}"
                )
            for name, (sizes, observed) in targets.items():
                results.setdefault((mean_function, name), []).append(
                    gp.evaluate(sizes, observed)
                )
        print(f"  finished repeat {rep + 1}/{n_repeats}", flush=True)

    print(f"\nMean over {n_repeats} repeats:")
    for (mean_function, name), runs in results.items():
        avg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        print(
            f"  {mean_function:>9} {name:>10}: RMSE={avg['rmse']:.4f}  "
            f"quantized lik.={avg['quantized_likelihood']:.3f}  "
            f"95% coverage={avg['coverage']:.2f}"
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
    model = APExGP(dataset, n_seeds=2, seed=seed)
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
