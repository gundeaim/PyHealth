"""APEx-GP on EEGBCI motor imagery: learning-curve extrapolation and ablations.

PURPOSE
    Tests whether APEx-GP (Harvey, Chen, Kent & Hughes, ML4H 2023) can predict
    the test AUROC that a classifier will reach on a full training set from
    AUROCs measured on small "pilot" subsets, when applied to EEG instead of
    the paper's images. Task: rest (T0) vs imagined movement (T1/T2) in the
    PhysioNet EEG Motor Movement/Imagery dataset (EEGBCI), imagery runs 4, 6,
    8, 10, 12, 14. Features: 16 channels x 5 log band powers = 80 per 2 s
    window (``eegbci_features.build_feature_matrix``). Classifier: L2 logistic
    regression with C tuned on validation AUROC.

PROTOCOL (one trial per seed; every configuration uses the same seeds)
    1. Split subjects 60/20/20 into train/val/test (subject-disjoint).
    2. Draw nested pilot subsets of the training split (default sizes
       60, 94, 147, 230, 360 windows, as in the paper).
    3. Train a classifier on each subset and record its test AUROC.
    4. MAP-fit APExGPRegressor to (pilot sizes, pilot AUROCs).
    5. Predict the mean and 95% HDI of AUROC at N_full = size of the whole
       training split.
    6. Train a classifier on the whole training split (same val/test) and
       record its test AUROC as the observed value.
    Configurations that share a seed and a pilot schedule reuse the same
    classifiers (steps 1-3 and 6), so they differ only in the GP.

ABLATIONS (--ablations; the baseline row is always included)
    baseline        paper sizes, power-law mean, data-dependent sigma prior.
    A_arctan        arctan mean instead of power law.
    B_small         pilot sizes 30, 47, 74, 115, 180 (half the paper's).
    B_large         pilot sizes 120, 188, 294, 460, 720 (double).
    C_fixed_prior   fixed sigma ~ TruncNormal(0, inf, 0.01, 0.02) instead of
                    the authors' data-dependent prior. Tests whether the
                    data-dependent prior is what keeps intervals calibrated.

POINT-ESTIMATE BASELINES (always added; the paper compares against these)
    LS_power_law, LS_arctan: the GP mean functions fitted by least squares to
    the baseline's pilot AUROCs (same seeds, same classifiers, same epsilon
    constraint), with no uncertainty. Their coverage/QL/width are "n/a".

HOW TO READ THE RESULTS
    The Markdown table has one row per configuration, aggregated over seeds:
      RMSE          sqrt(mean over seeds of (predicted mean - observed)^2).
      |err| ± sd    mean and std across seeds of |predicted - observed|.
      bias          mean of (predicted - observed); < 0 means the GP is
                    pessimistic about what more data buys.
      coverage      fraction of seeds whose observed AUROC lies in the 95%
                    HDI. Well calibrated means close to 0.95; with 10 seeds,
                    one miss moves it by 0.10, so treat small differences as
                    noise.
      QL            mean quantized likelihood, the predictive probability of
                    (observed - 0.01, observed + 0.01); higher is better.
      width         mean 95% HDI width; coverage must be read together with
                    width (very wide intervals cover trivially).
      ε at upper / lower
                    fraction of seeds whose fitted epsilon lies within 1% of
                    its prior support [epsilon_min, 1 - max pilot AUROC] of
                    the upper bound (ceiling = best pilot AUROC: the curve is
                    flat at the pilot level) or the lower bound (ceiling near
                    1). High rates flag extrapolations driven by the bound.
    The CSV has one row per (configuration, seed) with the pilot AUROCs, the
    prediction, the observed value, the ceiling diagnostic (max_pilot_auroc,
    fitted_ceiling, pred_minus_max_pilot, epsilon_at_bound), fitted GP
    hyperparameters and timings. GP-only columns are empty for LS rows.

USAGE
    Quick check, synthetic data, no downloads (a few minutes):
        python examples/eegbci_motor_imagery_apex_gp.py --synthetic
    Quick real run on 20 subjects, cached features:
        python examples/eegbci_motor_imagery_apex_gp.py --subjects 1-20 \\
            --data-root ~/eegbci_data --cache eegbci_1-20.npz
    Full run (109 subjects, 10 seeds, all ablations):
        python examples/eegbci_motor_imagery_apex_gp.py --data-root ~/eegbci_data \\
            --cache eegbci_all.npz
    A full run fits 5 configurations x 10 seeds = 50 GPs (up to ~20k Adam
    iterations each at the authors' lr=0.01) and 4 x 10 x 6 classifiers;
    lower --gp-iterations for a faster, less converged run.

RESULTS (full run on all 109 subjects, 10 seeds)
    Setup: 109/109 subjects loaded, 38,895 windows x 80 features (19,220 rest /
    19,675 imagery); GP up to 20,000 Adam iterations, 2 restarts; 34.2 min.
    N_full = training split = ~23,160 windows per seed, about 64x the largest
    paper pilot (360). Mean observed AUROC at N_full: 0.622 (same in all rows).
    baseline, A and C share each seed's classifiers (identical pilot and
    observed AUROCs), so their differences come from the GP alone.

    config          pilots    RMSE   |err| ± sd      bias     cov   QL     width
    baseline        60-360    0.042  0.035 ± 0.025   +0.018   0.90  0.142  0.130
    A_arctan        60-360    0.213  0.167 ± 0.140   +0.159   0.40  0.087  0.121
    B_small         30-180    0.058  0.051 ± 0.029   -0.015   0.80  0.096  0.142
    B_large         120-720   0.030  0.022 ± 0.021   +0.010   0.90  0.190  0.120
    C_fixed_prior   60-360    0.040  0.031 ± 0.026   +0.018   0.50  0.186  0.054

    1. Baseline: 64x extrapolation with RMSE 0.042 and a slight optimistic
       bias (+0.018). It covered 9 of 10 seeds, consistent with nominal 95%
       (10 seeds cannot tell 0.90 from 0.95), but at a cost: the 95% intervals
       average 0.13 wide.
    2. Arctan mean (A) is unreliable here: in 6 of 10 seeds it predicted
       0.80-0.96 against an observed ~0.62, giving 5x the baseline RMSE and
       0.40 coverage (9/10 vs 4/10 covered, Fisher exact p = 0.06). The two
       means agreed on a 20-subject pilot run, so the failure depends on the
       data rather than appearing always. We did not diagnose the cause;
       inspect fitted_epsilon in the CSV (a near-0 epsilon means the fit
       saturates near AUROC 1).
    3. Pilot schedule (B): RMSE falls monotonically with pilot size (0.058 ->
       0.042 -> 0.030) and quantized likelihood rises (0.096 -> 0.142 ->
       0.190), with similar interval widths. The coverage differences
       (0.80/0.90/0.90) are one miss and may be noise.
    4. Fixed sigma prior (C): point predictions barely change (RMSE 0.040 vs
       0.042, same bias), but intervals are 2.4x narrower (0.054 vs 0.130) and
       covered only 5 of 10 seeds vs 9 (Fisher exact p = 0.14). That is
       suggestive rather than conclusive at 10 seeds. C's higher QL (0.186)
       reflects sharper, not better-calibrated, intervals.
    5. Why C differs on real but not synthetic data: the authors' prior sets
       sigma's prior so that 6 * sqrt(sigma^2 + tau^2) spans 1/2 to 3/4 of the
       headroom (1 - epsilon_min - max pilot AUROC). With pilot AUROCs near
       0.6 the headroom is ~0.4, the targets are ~0.19-0.29, and the
       calibration picks sigma ~ TruncNormal(~0.04, ~0.01), about 4x the fixed
       prior's loc, which allows wide intervals. On the synthetic data (max
       pilot ~0.95) the targets of 0.025-0.038 fall below the spread that tau's
       prior alone implies (6 * tau ~ 0.06-0.12, tau >= 0.01), so the grid
       search returns its smallest cell (0.001, 0.001); sigma is then
       negligible under either prior and the tau floor sets the width. The
       data-dependent prior therefore matters most when the learning curve is
       far from its ceiling. The exact prior used per trial is in the CSV
       columns sigma_prior_loc and sigma_prior_scale.

    Limitations: each seed has a single subject split and test set, so the
    observed AUROC is itself noisy; windows come in correlated pairs from the
    same ~4 s trial; cross-subject EEG gives a low ceiling (~0.62), leaving
    little curve to extrapolate. This is our extension of APEx-GP to non-image
    data, not a reproduction of the paper's own datasets.

NOTES
    The protocol function ``run_protocol(X, y, groups, config, seeds)`` is
    data-agnostic: any feature matrix with binary labels and group IDs works
    (e.g. ChestX-ray14 ViT embeddings with patient IDs as groups).
    Windows come in pairs from the same ~4 s EEGBCI trial, so N windows carry
    roughly N/2 trials' worth of information; keep this in mind when comparing
    sizes with the paper's image counts.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pyhealth.models.apex_gp import (
    DEFAULT_C_GRID,
    DEFAULT_N_ITERATIONS,
    DEFAULT_SUBSET_SIZES,
    APExGPPriors,
    APExGPRegressor,
    draw_nested_subsets,
    epsilon_at_bound,
    fit_learning_curve_least_squares,
    fit_logistic_regression,
    split_by_group,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from eegbci_features import (  # noqa: E402
    MOTOR_IMAGERY_RUNS,
    build_feature_matrix,
    make_synthetic_samples,
)

logger = logging.getLogger("apex_gp_experiment")

PAPER_SIZES: tuple[int, ...] = tuple(DEFAULT_SUBSET_SIZES)
SMALL_SIZES: tuple[int, ...] = (30, 47, 74, 115, 180)
LARGE_SIZES: tuple[int, ...] = (120, 188, 294, 460, 720)
FIXED_SIGMA_PRIOR = APExGPPriors(sigma_loc=0.01, sigma_scale=0.02)
ALL_EEGBCI_SUBJECTS: tuple[int, ...] = tuple(range(1, 110))

# Synthetic data: amplitudes chosen so test AUROC rises gradually from ~0.73
# (30 windows) to ~0.91 (full training split) instead of saturating at 1.0.
SYNTHETIC_KWARGS = {
    "n_subjects": 20,
    "runs": tuple(sorted(MOTOR_IMAGERY_RUNS)),
    "windows_per_run": 60,
    "rest_mu_amplitude": 0.52,
    "imagery_mu_amplitude": 0.48,
    "subject_variability": 0.2,
}


# =========================================================================== #
# Configuration and results
# =========================================================================== #
@dataclass(frozen=True)
class ExperimentConfig:
    """One row of the results table.

    Attributes:
        name: Short identifier shown in the table and CSV.
        pilot_sizes: Strictly increasing pilot subset sizes (samples).
        mean_function: GP mean, "power_law" or "arctan".
        priors: GP priors (default: authors' data-dependent sigma prior).
        split_fractions: Train/val/test fractions of groups.
        c_grid: Candidate logistic regression C values.
        gp_lr: Adam learning rate for the GP.
        gp_iterations: Maximum Adam iterations per restart.
        gp_restarts: Number of GP optimizer restarts.
        level: Predictive interval mass.
        delta: Half-width for the quantized likelihood.
        description: Human-readable summary for the table.
    """

    name: str
    pilot_sizes: tuple[int, ...] = PAPER_SIZES
    mean_function: str = "power_law"
    priors: APExGPPriors = field(default_factory=APExGPPriors)
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    c_grid: tuple[float, ...] = tuple(DEFAULT_C_GRID)
    gp_lr: float = 0.01
    gp_iterations: int = DEFAULT_N_ITERATIONS
    gp_restarts: int = 2
    level: float = 0.95
    delta: float = 0.01
    description: str = ""


@dataclass(frozen=True)
class LearningCurve:
    """Classifier results for one seed and pilot schedule (protocol steps 1-3, 6).

    Attributes:
        seed: Trial seed.
        pilot_sizes: Pilot subset sizes.
        pilot_aurocs: Test AUROC at each pilot size.
        pilot_best_c: Selected C at each pilot size.
        n_full: Size of the whole training split.
        observed_auroc: Test AUROC of the classifier trained on all of it.
        full_best_c: Selected C for that classifier.
        n_train_groups: Number of training groups (subjects/patients).
        seconds: Wall time spent training classifiers.
    """

    seed: int
    pilot_sizes: tuple[int, ...]
    pilot_aurocs: tuple[float, ...]
    pilot_best_c: tuple[float, ...]
    n_full: int
    observed_auroc: float
    full_best_c: float
    n_train_groups: int
    seconds: float


@dataclass(frozen=True)
class TrialResult:
    """One (configuration, seed) outcome; one CSV row.

    GP-only fields are None for least-squares baseline rows (empty in the CSV).

    Attributes:
        config: Configuration name.
        seed: Trial seed.
        method: "gp" or "least_squares".
        mean_function: Learning-curve mean function.
        pilot_sizes: Pilot sizes, ";"-joined.
        pilot_aurocs: Pilot test AUROCs, ";"-joined.
        n_full: Size of the whole training split.
        predicted_mean: Point prediction at n_full (GP: truncated mean).
        observed: Test AUROC of the classifier trained on n_full samples.
        error: predicted_mean - observed.
        abs_error: |error|.
        max_pilot_auroc: Largest pilot AUROC, y' (epsilon <= 1 - y').
        fitted_epsilon: Fitted saturation gap.
        fitted_ceiling: 1 - fitted_epsilon, the curve's asymptote.
        pred_minus_max_pilot: predicted_mean - max_pilot_auroc.
        epsilon_at_bound: "upper", "lower" or "none" (within 1% of the
            epsilon support width of a bound).
        classifier_seconds: Wall time of the classifiers for this seed.
        gp_seconds: Wall time of the curve fit and prediction (GP or LS).
        lower: Lower bound of the HDI (GP only).
        upper: Upper bound of the HDI (GP only).
        covered: Whether observed lies in [lower, upper] (GP only).
        interval_width: upper - lower (GP only).
        quantized_likelihood: Predictive mass on observed +/- delta (GP only).
        sigma_prior_loc: Location of the sigma prior used (GP only).
        sigma_prior_scale: Scale of the sigma prior used (GP only).
        fitted_sigma: Fitted kernel output scale (GP only).
        fitted_lengthscale: Fitted kernel lengthscale, log10 units (GP only).
        fitted_tau: Fitted noise standard deviation (GP only).
        gp_iterations_run: Adam iterations per restart, ";"-joined (GP only).
    """

    config: str
    seed: int
    method: str
    mean_function: str
    pilot_sizes: str
    pilot_aurocs: str
    n_full: int
    predicted_mean: float
    observed: float
    error: float
    abs_error: float
    max_pilot_auroc: float
    fitted_epsilon: float
    fitted_ceiling: float
    pred_minus_max_pilot: float
    epsilon_at_bound: str
    classifier_seconds: float
    gp_seconds: float
    lower: float | None = None
    upper: float | None = None
    covered: bool | None = None
    interval_width: float | None = None
    quantized_likelihood: float | None = None
    sigma_prior_loc: float | None = None
    sigma_prior_scale: float | None = None
    fitted_sigma: float | None = None
    fitted_lengthscale: float | None = None
    fitted_tau: float | None = None
    gp_iterations_run: str | None = None


CurveCache = dict[tuple, LearningCurve]


# =========================================================================== #
# Protocol (data-agnostic)
# =========================================================================== #
def _test_auroc(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    c_grid: Sequence[float],
    seed: int,
) -> tuple[float, float]:
    """Trains a validation-tuned classifier and returns (test AUROC, best C).

    Args:
        features: Feature matrix, shape (N, d).
        labels: Binary labels, shape (N,).
        train_idx: Training indices.
        val_idx: Validation indices.
        test_idx: Test indices.
        c_grid: Candidate C values.
        seed: Solver random state.

    Returns:
        A tuple ``(auroc, best_c)``.
    """
    model, best_c = fit_logistic_regression(
        features[train_idx],
        labels[train_idx],
        features[val_idx],
        labels[val_idx],
        c_grid,
        seed,
    )
    scores = model.predict_proba(features[test_idx])[:, 1]
    return float(roc_auc_score(labels[test_idx], scores)), best_c


def measure_learning_curve(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    pilot_sizes: Sequence[int],
    seed: int,
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> LearningCurve:
    """Protocol steps 1-3 and 6: pilot AUROCs and the observed full-data AUROC.

    Args:
        features: Feature matrix, shape (N, d).
        labels: Binary labels, shape (N,).
        groups: Group ID per sample (subject/patient), shape (N,).
        pilot_sizes: Strictly increasing pilot sizes.
        seed: Seed for the split and the subset draw.
        split_fractions: Train/val/test fractions of groups.
        c_grid: Candidate C values.

    Returns:
        The ``LearningCurve`` for this seed.

    Raises:
        ValueError: If val/test lack a class or the training split is not
            larger than the largest pilot size.
    """
    start = time.perf_counter()
    train, val, test = split_by_group(groups, split_fractions, seed)
    for name, idx in (("validation", val), ("test", test)):
        if len(np.unique(labels[idx])) < 2:
            raise ValueError(f"Seed {seed}: {name} split has a single class.")
    if len(train) <= max(pilot_sizes):
        raise ValueError(
            f"Seed {seed}: training split ({len(train)} samples) must exceed the "
            f"largest pilot size ({max(pilot_sizes)}); use more groups."
        )
    subsets = draw_nested_subsets(labels, train, pilot_sizes, seed)
    pilot = [
        _test_auroc(features, labels, idx, val, test, c_grid, seed) for idx in subsets
    ]
    observed, full_c = _test_auroc(features, labels, train, val, test, c_grid, seed)
    return LearningCurve(
        seed=seed,
        pilot_sizes=tuple(int(s) for s in pilot_sizes),
        pilot_aurocs=tuple(a for a, _ in pilot),
        pilot_best_c=tuple(c for _, c in pilot),
        n_full=len(train),
        observed_auroc=observed,
        full_best_c=full_c,
        n_train_groups=len(np.unique(groups[train])),
        seconds=time.perf_counter() - start,
    )


def _shared_fields(
    curve: LearningCurve,
    config_name: str,
    method: str,
    mean_function: str,
    predicted: float,
    epsilon: float,
    epsilon_min: float,
    seconds: float,
) -> dict[str, float | int | str]:
    """Fields common to GP and least-squares rows, incl. the ceiling diagnostic.

    Args:
        curve: Classifier results for one seed.
        config_name: Row name.
        method: "gp" or "least_squares".
        mean_function: Learning-curve mean function.
        predicted: Point prediction at ``curve.n_full``.
        epsilon: Fitted saturation gap.
        epsilon_min: Lower bound of epsilon.
        seconds: Fit wall time.

    Returns:
        Keyword arguments for ``TrialResult``.
    """
    max_pilot = float(max(curve.pilot_aurocs))
    error = predicted - curve.observed_auroc
    return {
        "config": config_name,
        "seed": curve.seed,
        "method": method,
        "mean_function": mean_function,
        "pilot_sizes": ";".join(str(s) for s in curve.pilot_sizes),
        "pilot_aurocs": ";".join(f"{a:.5f}" for a in curve.pilot_aurocs),
        "n_full": curve.n_full,
        "predicted_mean": predicted,
        "observed": curve.observed_auroc,
        "error": error,
        "abs_error": abs(error),
        "max_pilot_auroc": max_pilot,
        "fitted_epsilon": epsilon,
        "fitted_ceiling": 1.0 - epsilon,
        "pred_minus_max_pilot": predicted - max_pilot,
        "epsilon_at_bound": epsilon_at_bound(epsilon, max_pilot, epsilon_min),
        "classifier_seconds": curve.seconds,
        "gp_seconds": seconds,
    }


def fit_and_score_gp(curve: LearningCurve, config: ExperimentConfig) -> TrialResult:
    """Protocol steps 4, 5 and 7: fit the GP and score its prediction at N_full.

    Args:
        curve: Classifier results for one seed.
        config: Configuration (GP settings).

    Returns:
        The ``TrialResult`` for this configuration and seed.
    """
    start = time.perf_counter()
    gp = APExGPRegressor(
        mean_function=config.mean_function,
        priors=config.priors,
        lr=config.gp_lr,
        n_iterations=config.gp_iterations,
        n_restarts=config.gp_restarts,
        seed=curve.seed,
    )
    gp.fit(np.asarray(curve.pilot_sizes, float), np.asarray(curve.pilot_aurocs))
    mean, lower, upper = gp.predict_interval([curve.n_full], level=config.level)
    metrics = gp.evaluate(
        [curve.n_full], [curve.observed_auroc], delta=config.delta, level=config.level
    )
    hp = gp.hyperparameters_
    predicted, lo, hi = float(mean[0]), float(lower[0]), float(upper[0])
    shared = _shared_fields(
        curve,
        config.name,
        "gp",
        config.mean_function,
        predicted,
        hp.epsilon,
        config.priors.epsilon_min,
        time.perf_counter() - start,
    )
    return TrialResult(
        **shared,
        lower=lo,
        upper=hi,
        covered=bool(lo <= curve.observed_auroc <= hi),
        interval_width=hi - lo,
        quantized_likelihood=metrics["quantized_likelihood"],
        sigma_prior_loc=float(gp.sigma_prior_loc_),
        sigma_prior_scale=float(gp.sigma_prior_scale_),
        fitted_sigma=hp.sigma,
        fitted_lengthscale=hp.lengthscale,
        fitted_tau=hp.tau,
        gp_iterations_run=";".join(str(n) for n in gp.n_iterations_run_),
    )


def fit_and_score_least_squares(
    curve: LearningCurve, mean_function: str, name: str, epsilon_min: float = 0.0
) -> TrialResult:
    """Least-squares point-prediction baseline on the same learning curve.

    Args:
        curve: Classifier results for one seed (shared with the GP rows).
        mean_function: "power_law" or "arctan".
        name: Row name, e.g. "LS_power_law".
        epsilon_min: Lower bound of epsilon.

    Returns:
        A ``TrialResult`` whose interval/likelihood fields are None.
    """
    start = time.perf_counter()
    fit = fit_learning_curve_least_squares(
        curve.pilot_sizes, curve.pilot_aurocs, mean_function, epsilon_min
    )
    predicted = float(fit.predict([curve.n_full])[0])
    return TrialResult(
        **_shared_fields(
            curve,
            name,
            "least_squares",
            mean_function,
            predicted,
            fit.epsilon,
            epsilon_min,
            time.perf_counter() - start,
        )
    )


def get_learning_curve(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: ExperimentConfig,
    seed: int,
    cache: CurveCache,
) -> LearningCurve:
    """Returns the cached learning curve for (seed, schedule), computing it once.

    Args:
        features: Feature matrix, shape (N, d).
        labels: Binary labels, shape (N,).
        groups: Group ID per sample, shape (N,).
        config: Supplies pilot sizes, split fractions and C grid.
        seed: Trial seed.
        cache: Shared cache, updated in place.

    Returns:
        The ``LearningCurve``.
    """
    key = (seed, config.pilot_sizes, config.split_fractions, config.c_grid)
    if key not in cache:
        cache[key] = measure_learning_curve(
            features,
            labels,
            groups,
            config.pilot_sizes,
            seed,
            config.split_fractions,
            config.c_grid,
        )
    return cache[key]


def run_protocol(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: ExperimentConfig,
    seeds: Iterable[int],
    curve_cache: CurveCache | None = None,
    progress: Callable[[str], None] = print,
) -> list[TrialResult]:
    """Runs the full evaluation protocol for one configuration over seeds.

    Data-agnostic: works for any features with binary labels and group IDs.
    Pass the same ``curve_cache`` across configurations to reuse classifier
    results for identical (seed, pilot sizes, split, C grid).

    Args:
        features: Feature matrix, shape (N, d).
        labels: Binary labels, shape (N,).
        groups: Group ID per sample, shape (N,).
        config: Experiment configuration.
        seeds: Trial seeds.
        curve_cache: Optional shared cache of learning curves.
        progress: Callable receiving one progress line per trial.

    Returns:
        One ``TrialResult`` per seed.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    groups = np.asarray(groups)
    cache = curve_cache if curve_cache is not None else {}
    results = []
    for seed in seeds:
        curve = get_learning_curve(features, labels, groups, config, seed, cache)
        trial = fit_and_score_gp(curve, config)
        results.append(trial)
        progress(
            f"  [{config.name}] seed {seed}: predicted {trial.predicted_mean:.3f} "
            f"[{trial.lower:.3f}, {trial.upper:.3f}] at N={trial.n_full}, observed "
            f"{trial.observed:.3f} ({'covered' if trial.covered else 'MISSED'}); "
            f"epsilon at bound: {trial.epsilon_at_bound}; GP {trial.gp_seconds:.1f}s"
        )
    return results


def run_least_squares_baselines(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    base_config: ExperimentConfig,
    seeds: Iterable[int],
    curve_cache: CurveCache | None = None,
    progress: Callable[[str], None] = print,
) -> list[TrialResult]:
    """Adds the LS_power_law and LS_arctan rows on the baseline's curves.

    Uses the learning curves of ``base_config`` (same seeds, pilot sizes and
    classifiers as the GP baseline), so no new classifiers are trained when
    the cache is shared.

    Args:
        features: Feature matrix, shape (N, d).
        labels: Binary labels, shape (N,).
        groups: Group ID per sample, shape (N,).
        base_config: Configuration whose pilot schedule and split are reused.
        seeds: Trial seeds.
        curve_cache: Optional shared cache of learning curves.
        progress: Callable receiving one progress line per trial.

    Returns:
        Two ``TrialResult`` rows per seed (power law, then arctan).
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    groups = np.asarray(groups)
    cache = curve_cache if curve_cache is not None else {}
    seeds = list(seeds)
    results = []
    for mean_function in ("power_law", "arctan"):
        name = f"LS_{mean_function}"
        for seed in seeds:
            curve = get_learning_curve(
                features, labels, groups, base_config, seed, cache
            )
            trial = fit_and_score_least_squares(
                curve, mean_function, name, base_config.priors.epsilon_min
            )
            results.append(trial)
            progress(
                f"  [{name}] seed {seed}: predicted {trial.predicted_mean:.3f} at "
                f"N={trial.n_full}, observed {trial.observed:.3f}; epsilon at "
                f"bound: {trial.epsilon_at_bound}"
            )
    return results


# =========================================================================== #
# Reporting
# =========================================================================== #
def _mean_or_none(values: Sequence[float | bool | None]) -> float | None:
    """Mean of the values, or None if any is None (metric not applicable)."""
    if any(v is None for v in values):
        return None
    return float(np.mean(values))


def summarize(
    results: Sequence[TrialResult],
) -> list[dict[str, float | str | int | None]]:
    """Aggregates trial results per configuration (in first-seen order).

    Args:
        results: Trial results, possibly from several configurations.

    Returns:
        One dict per configuration with the table's columns; interval metrics
        are None for least-squares rows.
    """
    order = list(dict.fromkeys(r.config for r in results))
    rows = []
    for name in order:
        trials = [r for r in results if r.config == name]
        err = np.array([t.error for t in trials])
        bounds = [t.epsilon_at_bound for t in trials]
        rows.append(
            {
                "config": name,
                "mean_function": trials[0].mean_function,
                "pilot_sizes": trials[0].pilot_sizes.replace(";", ","),
                "n_seeds": len(trials),
                "mean_n_full": float(np.mean([t.n_full for t in trials])),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "abs_error_mean": float(np.mean(np.abs(err))),
                "abs_error_std": float(np.std(np.abs(err), ddof=1))
                if len(err) > 1
                else 0.0,
                "bias": float(np.mean(err)),
                "coverage": _mean_or_none([t.covered for t in trials]),
                "quantized_likelihood": _mean_or_none(
                    [t.quantized_likelihood for t in trials]
                ),
                "interval_width": _mean_or_none([t.interval_width for t in trials]),
                "mean_observed": float(np.mean([t.observed for t in trials])),
                "epsilon_upper_rate": bounds.count("upper") / len(bounds),
                "epsilon_lower_rate": bounds.count("lower") / len(bounds),
            }
        )
    return rows


def _fmt(value: float | None, spec: str) -> str:
    """Formats a number, or "n/a" for None."""
    return "n/a" if value is None else format(value, spec)


def format_markdown(rows: Sequence[dict[str, float | str | int | None]]) -> str:
    """Formats ``summarize`` output as a Markdown table.

    Args:
        rows: Output of ``summarize``.

    Returns:
        The table as a string.
    """
    header = (
        "| config | mean fn | pilot sizes | seeds | N_full | observed | RMSE | "
        "\\|err\\| ± sd | bias | coverage | QL | width | eps at upper / lower |"
    )
    lines = [header, "|" + "---|" * 13]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['mean_function']} | {r['pilot_sizes']} | "
            f"{r['n_seeds']} | {r['mean_n_full']:.0f} | {r['mean_observed']:.3f} | "
            f"{r['rmse']:.4f} | {r['abs_error_mean']:.4f} ± {r['abs_error_std']:.4f} "
            f"| {r['bias']:+.4f} | {_fmt(r['coverage'], '.2f')} | "
            f"{_fmt(r['quantized_likelihood'], '.3f')} | "
            f"{_fmt(r['interval_width'], '.3f')} | "
            f"{r['epsilon_upper_rate']:.2f} / {r['epsilon_lower_rate']:.2f} |"
        )
    return "\n".join(lines)


def write_csv(results: Sequence[TrialResult], path: Path) -> None:
    """Writes per-trial results to CSV.

    Args:
        results: Trial results.
        path: Output path (parent directories are created).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


# =========================================================================== #
# Configurations
# =========================================================================== #
def build_configs(
    ablations: Iterable[str], gp_iterations: int, gp_restarts: int
) -> list[ExperimentConfig]:
    """Builds the baseline plus the requested ablation configurations.

    Args:
        ablations: Any of "A", "B", "C".
        gp_iterations: Maximum Adam iterations per GP restart.
        gp_restarts: Number of GP restarts.

    Returns:
        Configurations, baseline first.

    Raises:
        ValueError: On an unknown ablation letter.
    """
    common = {"gp_iterations": gp_iterations, "gp_restarts": gp_restarts}
    configs = [
        ExperimentConfig(
            name="baseline", description="paper sizes, power law, data prior", **common
        )
    ]
    wanted = {a.strip().upper() for a in ablations if a.strip()}
    unknown = wanted - {"A", "B", "C"}
    if unknown:
        raise ValueError(f"Unknown ablations {sorted(unknown)}; choose from A, B, C.")
    if "A" in wanted:
        configs.append(
            ExperimentConfig(
                name="A_arctan",
                mean_function="arctan",
                description="arctan mean",
                **common,
            )
        )
    if "B" in wanted:
        configs += [
            ExperimentConfig(
                name="B_small",
                pilot_sizes=SMALL_SIZES,
                description="half sizes",
                **common,
            ),
            ExperimentConfig(
                name="B_large",
                pilot_sizes=LARGE_SIZES,
                description="double sizes",
                **common,
            ),
        ]
    if "C" in wanted:
        configs.append(
            ExperimentConfig(
                name="C_fixed_prior",
                priors=FIXED_SIGMA_PRIOR,
                description="fixed sigma ~ TN(0.01, 0.02)",
                **common,
            )
        )
    return configs


# =========================================================================== #
# Data loading
# =========================================================================== #
def parse_subjects(spec: str) -> list[int]:
    """Parses a subject specification such as "1-20" or "1,3,5-9".

    Args:
        spec: Comma-separated subject numbers and inclusive ranges.

    Returns:
        Sorted unique subject numbers.

    Raises:
        ValueError: If the specification is malformed or out of 1-109.
    """
    subjects: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            subjects.update(range(lo, hi + 1))
        else:
            subjects.add(int(part))
    if not subjects or min(subjects) < 1 or max(subjects) > 109:
        raise ValueError(f"Subjects must be within 1-109; got {spec!r}.")
    return sorted(subjects)


def load_eegbci_features(
    subjects: Sequence[int],
    data_root: Path,
    runs: Sequence[int] = tuple(sorted(MOTOR_IMAGERY_RUNS)),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Loads EEGBCI subject by subject and builds band-power features.

    Each subject is loaded as its own ``EEGBCIDataset`` so a subject whose
    recordings fail to download, parse or featurize is skipped and logged
    instead of aborting the run.

    Args:
        subjects: Subject numbers (1-109).
        data_root: Download/cache root passed to ``EEGBCIDataset``.
        runs: Imagery runs to load.

    Returns:
        ``(X, y, groups, skipped_subjects)``.

    Raises:
        RuntimeError: If no subject could be loaded.
    """
    from pyhealth.datasets import EEGBCIDataset
    from pyhealth.tasks.eegbci import EEGMotorImageryEEGBCI

    parts, skipped = [], []
    for i, subject in enumerate(subjects, start=1):
        try:
            dataset = EEGBCIDataset(
                root=str(data_root), subjects=[subject], runs=list(runs), download=True
            )
            samples = list(dataset.set_task(EEGMotorImageryEEGBCI()))
            parts.append(build_feature_matrix(samples))
            print(
                f"Loaded subject {subject} ({i}/{len(subjects)}): "
                f"{len(parts[-1][1])} windows",
                flush=True,
            )
        except Exception as err:  # noqa: BLE001 - skip any failing subject
            logger.warning(
                "Skipping subject %d: %s: %s", subject, type(err).__name__, err
            )
            skipped.append(subject)
    if not parts:
        raise RuntimeError("No EEGBCI subject could be loaded.")
    features = np.vstack([p[0] for p in parts])
    labels = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([np.asarray(p[2]).astype(str) for p in parts])
    return features, labels, groups, skipped


def load_synthetic_features(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds features from ``make_synthetic_samples`` (no downloads).

    Args:
        seed: Seed for the synthetic signals.

    Returns:
        ``(X, y, groups)``.
    """
    samples = make_synthetic_samples(seed=seed, **SYNTHETIC_KWARGS)
    features, labels, groups = build_feature_matrix(samples)
    return features, labels, groups.astype(str)


def get_features(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, groups), using the .npz cache when it matches the request.

    Args:
        args: Parsed command-line arguments.

    Returns:
        ``(X, y, groups)``.
    """
    source = "synthetic" if args.synthetic else f"eegbci:{args.subjects}"
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if str(stored["source"]) == source:
            print(f"Loaded cached features from {cache} ({source})")
            return stored["X"], stored["y"], stored["groups"]
        print(f"Cache {cache} holds {stored['source']!s}, not {source}; rebuilding.")

    if args.synthetic:
        features, labels, groups = load_synthetic_features(args.base_seed)
    else:
        subjects = parse_subjects(args.subjects)
        features, labels, groups, skipped = load_eegbci_features(
            subjects, Path(args.data_root).expanduser()
        )
        if skipped:
            print(f"Skipped {len(skipped)} subject(s): {skipped}")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=features, y=labels, groups=groups, source=source)
        print(f"Saved features to {cache}")
    return features, labels, groups


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed namespace with mode-dependent defaults resolved.
    """
    parser = argparse.ArgumentParser(
        description="APEx-GP learning-curve extrapolation on EEGBCI motor imagery."
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="use synthetic EEG (no downloads)"
    )
    parser.add_argument(
        "--subjects", default="1-109", help='EEGBCI subjects, e.g. "1-20" or "1,3,5-9"'
    )
    parser.add_argument(
        "--seeds", type=int, default=None, help="trials per config (10; synthetic: 3)"
    )
    parser.add_argument("--base-seed", type=int, default=0, help="first trial seed")
    parser.add_argument(
        "--data-root", default=str(Path.home() / "eegbci_data"), help="EEGBCI root"
    )
    parser.add_argument("--cache", default=None, help="X/y/groups .npz cache path")
    parser.add_argument(
        "--ablations", default="A,B,C", help='comma-separated subset of "A,B,C", or ""'
    )
    parser.add_argument(
        "--gp-iterations",
        type=int,
        default=None,
        help=(
            f"max Adam iterations per restart ({DEFAULT_N_ITERATIONS}; synthetic: 4000)"
        ),
    )
    parser.add_argument("--gp-restarts", type=int, default=2, help="GP restarts")
    parser.add_argument(
        "--output",
        default="apex_gp_eegbci_trials.csv",
        help="per-trial CSV path (synthetic runs add a _synthetic suffix)",
    )
    args = parser.parse_args(argv)
    if args.seeds is None:
        args.seeds = 3 if args.synthetic else 10
    if args.gp_iterations is None:
        args.gp_iterations = 4000 if args.synthetic else DEFAULT_N_ITERATIONS
    if args.synthetic and args.output == parser.get_default("output"):
        args.output = "apex_gp_eegbci_trials_synthetic.csv"
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the experiment and prints the results table.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    features, labels, groups = get_features(args)
    n_groups = len(np.unique(groups))
    print(
        f"Data: {features.shape[0]} samples x {features.shape[1]} features, "
        f"{n_groups} groups, class balance {np.bincount(labels).tolist()}"
    )

    configs = build_configs(
        args.ablations.split(","), args.gp_iterations, args.gp_restarts
    )
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    print(
        f"Running {len(configs)} configuration(s) x {len(seeds)} seed(s); "
        f"GP max iterations {args.gp_iterations}, restarts {args.gp_restarts}."
    )
    cache: CurveCache = {}
    results: list[TrialResult] = []
    start = time.perf_counter()
    for i, config in enumerate(configs, start=1):
        print(f"\n[{i}/{len(configs)}] {config.name}: {config.description}", flush=True)
        results += run_protocol(
            features,
            labels,
            groups,
            config,
            seeds,
            curve_cache=cache,
            progress=lambda line: print(line, flush=True),
        )
    print("\nLeast-squares baselines on the baseline's learning curves", flush=True)
    results += run_least_squares_baselines(
        features,
        labels,
        groups,
        configs[0],
        seeds,
        curve_cache=cache,
        progress=lambda line: print(line, flush=True),
    )

    output = Path(args.output)
    write_csv(results, output)
    print(f"\nFinished in {(time.perf_counter() - start) / 60:.1f} min.")
    print(f"Per-trial results: {output.resolve()}\n")
    print(format_markdown(summarize(results)))


if __name__ == "__main__":
    main()