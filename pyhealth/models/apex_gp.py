"""APEx-GP: extrapolating classifier AUROC from small pilot datasets.

Implements the method of Harvey, Chen, Kent & Hughes, "A Probabilistic Method to
Predict Classifier Accuracy on Larger Datasets given Small Pilot Data" (ML4H 2023,
PMLR 225; arXiv:2311.18025).

A classifier is trained on nested pilot subsets of increasing size and its test
AUROC is recorded at each size. A Gaussian process with a learning-curve mean
function (power law or arctan) and a squared-exponential kernel on log10 size is
MAP-fitted to those points, and its truncated posterior predictive gives AUROC
forecasts with uncertainty at larger training-set sizes.

Parameterization, priors and optimizer settings follow the authors' public code
(tufts-ml/extrapolating-classifier-accuracy-to-larger-datasets), reimplemented
in plain torch without gpytorch.

The module has three layers, each usable on its own:

1. GP core (torch): mean functions, kernel, marginal likelihood, posterior
   predictive and the ``APExGPRegressor`` that MAP-fits them.
2. Evaluation (NumPy/SciPy): truncated-normal mean, highest-density interval,
   RMSE, quantized likelihood and coverage.
3. Pilot experiment (scikit-learn) and the ``APExGP`` PyHealth model wrapper.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ..datasets import SampleDataset
from .base_model import BaseModel

logger = logging.getLogger(__name__)

MeanFunctionName = Literal["power_law", "arctan"]

DEFAULT_SUBSET_SIZES: tuple[int, ...] = (60, 94, 147, 230, 360)
"""Pilot training-set sizes used in the paper."""

DEFAULT_C_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
"""Inverse L2 regularization strengths searched on the validation split."""

SIGMA_PRIOR_GRID: np.ndarray = np.linspace(0.001, 0.05, 100)
"""Candidate loc and scale values searched by ``calibrate_sigma_prior``."""

DEFAULT_N_ITERATIONS = 20_000
"""Maximum Adam iterations per restart at lr=0.01. From the authors' raw = 0
initialization, sigma and tau start near 0.83 and must shrink to ~0.005-0.01,
which takes ~16k iterations at lr=0.01; the data-driven restart converges in
~2-4k and stops early."""

NOISE_VARIANCE_FLOOR = 1e-4
"""Lower bound on tau**2 (gpytorch GaussianLikelihood default), so tau >= 0.01."""

_LOG_2PI = math.log(2.0 * math.pi)
_LENGTHSCALE_FLOOR = 1e-6  # keeps (log10 x - log10 x') / lengthscale finite
_PARAM_NAMES = ("epsilon", "theta1", "theta2", "sigma", "lengthscale", "tau")


# =========================================================================== #
# 1. GP core (torch)
# =========================================================================== #
def power_law_mean(
    x: torch.Tensor,
    epsilon: torch.Tensor | float,
    theta1: torch.Tensor | float,
    theta2: torch.Tensor | float,
) -> torch.Tensor:
    """Power-law learning-curve mean, m(x) = (1 - eps) - theta1 * x**theta2.

    Args:
        x: Training-set sizes, shape (n,).
        epsilon: Saturation gap; m(x) -> 1 - epsilon as x -> infinity.
        theta1: Non-negative scale.
        theta2: Exponent in [-1, 0].

    Returns:
        Mean AUROC at each size, shape (n,).
    """
    return (1.0 - epsilon) - theta1 * torch.pow(x, theta2)


def arctan_mean(
    x: torch.Tensor,
    epsilon: torch.Tensor | float,
    theta1: torch.Tensor | float,
    theta2: torch.Tensor | float,
) -> torch.Tensor:
    """Arctan learning-curve mean, m(x) = (2/pi) atan(theta1 pi/2 x + theta2) - eps.

    Args:
        x: Training-set sizes, shape (n,).
        epsilon: Saturation gap; m(x) -> 1 - epsilon as x -> infinity.
        theta1: Non-negative slope.
        theta2: Non-negative offset.

    Returns:
        Mean AUROC at each size, shape (n,).
    """
    inner = theta1 * (math.pi / 2.0) * x + theta2
    return (2.0 / math.pi) * torch.atan(inner) - epsilon


MEAN_FUNCTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "power_law": power_law_mean,
    "arctan": arctan_mean,
}


def log_rbf_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    sigma: torch.Tensor | float,
    lengthscale: torch.Tensor | float,
) -> torch.Tensor:
    """Squared-exponential kernel on base-10 log sizes.

    k(x, x') = sigma**2 * exp(-(log10 x - log10 x')**2 / (2 * lengthscale**2)),
    matching the authors' code, which feeds log10(x) to an RBF kernel.

    Args:
        x1: Positive sizes, shape (n,).
        x2: Positive sizes, shape (m,).
        sigma: Output scale (> 0); sigma**2 is gpytorch's ``outputscale``.
        lengthscale: Lengthscale in log10-size units (> 0).

    Returns:
        Kernel matrix, shape (n, m).
    """
    diff = torch.log10(x1).unsqueeze(-1) - torch.log10(x2).unsqueeze(-2)
    return sigma**2 * torch.exp(-0.5 * (diff / lengthscale) ** 2)


def stable_cholesky(
    matrix: torch.Tensor, jitter: float = 1e-8, max_tries: int = 8
) -> torch.Tensor:
    """Lower Cholesky factor, adding increasing diagonal jitter until it succeeds.

    Args:
        matrix: Symmetric positive (semi-)definite matrix, shape (n, n).
        jitter: Initial diagonal jitter; multiplied by 10 on each failure.
        max_tries: Maximum number of attempts.

    Returns:
        Lower-triangular factor L with L @ L.T = matrix + jitter * I.

    Raises:
        RuntimeError: If no attempt yields a valid factorization.
    """
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    for attempt in range(max_tries):
        chol, info = torch.linalg.cholesky_ex(matrix + jitter * 10**attempt * eye)
        if not bool(info.any()):
            return chol
    raise RuntimeError(
        f"Cholesky failed after {max_tries} attempts (max jitter "
        f"{jitter * 10 ** (max_tries - 1):.1e})."
    )


def gp_log_marginal_likelihood(
    y: torch.Tensor,
    mean: torch.Tensor,
    kernel: torch.Tensor,
    tau: torch.Tensor | float,
    jitter: float = 1e-8,
) -> torch.Tensor:
    """Gaussian log marginal likelihood with the latent f integrated out.

    log p(y) = -R/2 log 2pi - 1/2 log|K + tau^2 I|
               - 1/2 (y - m)^T (K + tau^2 I)^{-1} (y - m).

    Args:
        y: Observed AUROCs, shape (R,).
        mean: Prior mean at the observed sizes, shape (R,).
        kernel: Kernel matrix K, shape (R, R).
        tau: Observation noise standard deviation.
        jitter: Initial Cholesky jitter.

    Returns:
        Scalar log marginal likelihood.
    """
    n = y.shape[-1]
    eye = torch.eye(n, dtype=y.dtype, device=y.device)
    chol = stable_cholesky(kernel + tau**2 * eye, jitter)
    resid = (y - mean).unsqueeze(-1)
    whitened = torch.linalg.solve_triangular(chol, resid, upper=False)
    log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()
    return -0.5 * n * _LOG_2PI - 0.5 * log_det - 0.5 * whitened.pow(2).sum()


def gp_posterior_predictive(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    mean_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma: torch.Tensor | float,
    lengthscale: torch.Tensor | float,
    tau: torch.Tensor | float,
    jitter: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior predictive of noisy observations y* at new sizes.

    mu    = m* + K*^T (K + tau^2 I)^{-1} (y - m)
    Sigma = K** + tau^2 I - K*^T (K + tau^2 I)^{-1} K*

    Args:
        x_train: Observed sizes, shape (R,).
        y_train: Observed AUROCs, shape (R,).
        x_test: Query sizes, shape (Q,).
        mean_fn: Prior mean function of sizes.
        sigma: Kernel output scale.
        lengthscale: Kernel lengthscale (log10-size units).
        tau: Observation noise standard deviation.
        jitter: Initial Cholesky jitter.

    Returns:
        A tuple ``(mean, cov)`` of shapes (Q,) and (Q, Q), untruncated.
    """
    k_tt = log_rbf_kernel(x_train, x_train, sigma, lengthscale)
    k_ts = log_rbf_kernel(x_train, x_test, sigma, lengthscale)
    k_ss = log_rbf_kernel(x_test, x_test, sigma, lengthscale)
    eye_r = torch.eye(x_train.shape[-1], dtype=k_tt.dtype, device=k_tt.device)
    eye_q = torch.eye(x_test.shape[-1], dtype=k_tt.dtype, device=k_tt.device)

    chol = stable_cholesky(k_tt + tau**2 * eye_r, jitter)
    v = torch.linalg.solve_triangular(chol, k_ts, upper=False)  # (R, Q)
    resid = (y_train - mean_fn(x_train)).unsqueeze(-1)
    w = torch.linalg.solve_triangular(chol, resid, upper=False)  # (R, 1)
    mean = mean_fn(x_test) + (v.transpose(-1, -2) @ w).squeeze(-1)
    cov = k_ss + tau**2 * eye_q - v.transpose(-1, -2) @ v
    return mean, cov


def truncated_normal_log_prob(
    value: torch.Tensor, loc: float, scale: float
) -> torch.Tensor:
    """Log density of Normal(loc, scale) truncated to (0, infinity).

    Args:
        value: Positive value(s).
        loc: Location of the untruncated normal (may be negative).
        scale: Scale of the untruncated normal (> 0).

    Returns:
        Log density with the same shape as ``value``.
    """
    z = (value - loc) / scale
    log_mass = torch.special.log_ndtr(
        torch.as_tensor(loc / scale, dtype=value.dtype, device=value.device)
    )
    return -0.5 * z**2 - math.log(scale) - 0.5 * _LOG_2PI - log_mass


def _sample_positive_truncnorm(
    loc: float | np.ndarray,
    scale: float | np.ndarray,
    size: int | tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draws Normal(loc, scale) samples truncated to (0, infinity).

    Uses the inverse CDF written through the upper tail,
    x = loc - scale * Phi^{-1}((1 - u) * Phi(loc / scale)) with u ~ U[0, 1),
    which is accurate for any sign of ``loc``. ``loc`` and ``scale``
    broadcast against ``size``.

    Args:
        loc: Location(s) of the untruncated normal.
        scale: Scale(s) of the untruncated normal (> 0).
        size: Output shape.
        rng: NumPy random generator.

    Returns:
        Positive samples of shape ``size``.
    """
    loc = np.asarray(loc, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    upper_mass = ndtr(loc / scale)  # P(X > 0) for the untruncated normal
    return loc - scale * ndtri((1.0 - rng.random(size)) * upper_mass)


def calibrate_sigma_prior(
    tau_loc: float,
    tau_scale: float,
    desired_low: float,
    desired_high: float,
    num_samples: int = 1000,
    grid: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Chooses the sigma prior so the implied predictive spread hits a target.

    Reimplements ``calc_outputscale_prior`` from the authors' ``priors.py``.
    Draws ``num_samples`` values of tau from its prior once. Then, for every
    (m, s) on ``grid`` x ``grid`` (m outer, s inner), draws ``num_samples``
    values of sigma ~ TruncNormal(0, inf, m, s), computes
    6 * sqrt(sigma**2 + tau**2), and scores the 20th/50th/80th percentiles
    against (low, (low + high) / 2, high) by summed absolute error. The first
    (m, s) achieving the smallest error wins, as with the original strict "<".
    A seeded generator makes the result deterministic.

    Args:
        tau_loc: Location of the tau prior.
        tau_scale: Scale of the tau prior.
        desired_low: Target 20th percentile of 6 * sqrt(sigma**2 + tau**2).
        desired_high: Target 80th percentile.
        num_samples: Monte Carlo samples per grid cell (authors: 1000).
        grid: Candidate values for both m and s. Defaults to
            ``SIGMA_PRIOR_GRID``, the authors' ``np.linspace(0.001, 0.05, 100)``.
        seed: Seed for the Monte Carlo draws.

    Returns:
        The chosen ``(loc, scale)`` of the sigma prior.
    """
    grid = SIGMA_PRIOR_GRID if grid is None else np.asarray(grid, np.float64)
    rng = np.random.default_rng(seed)
    tau = _sample_positive_truncnorm(tau_loc, tau_scale, num_samples, rng)
    desired = np.array([desired_low, 0.5 * (desired_low + desired_high), desired_high])
    best_dist, best_m, best_s = math.inf, 0.0, 0.0
    for m in grid:
        sigma = _sample_positive_truncnorm(
            m, grid[:, None], (len(grid), num_samples), rng
        )
        spread = 6.0 * np.sqrt(sigma**2 + tau**2)
        quantiles = np.percentile(spread, [20, 50, 80], axis=1).T  # (n_s, 3)
        dist = np.abs(quantiles - desired).sum(axis=1)
        j = int(np.argmin(dist))
        if dist[j] < best_dist:
            best_dist, best_m, best_s = float(dist[j]), float(m), float(grid[j])
    return best_m, best_s


@dataclass(frozen=True)
class APExGPPriors:
    """Priors on the GP hyperparameters eta = {tau, sigma, lengthscale, epsilon}.

    Lengthscale, sigma and tau get Normal(loc, scale) priors truncated to
    (0, infinity); epsilon gets Uniform(epsilon_min, 1 - max observed AUROC).
    The mean-function parameters theta have no prior, only box constraints.

    By default the sigma prior is data-dependent, as in the authors' code: at
    fit time ``calibrate_sigma_prior`` picks (loc, scale) so that the 20th and
    80th percentiles of 6 * sqrt(sigma**2 + tau**2) under the priors land near
    ``sigma_low_fraction`` and ``sigma_high_fraction`` times the headroom
    (1 - epsilon_min) - max(y). Set both ``sigma_loc`` and ``sigma_scale`` to
    use a fixed prior instead.

    Attributes:
        lengthscale_loc: Lengthscale prior location, in log10-size units.
        lengthscale_scale: Lengthscale prior scale. The authors' code uses
            2.12; the paper's appendix reports 2.14.
        tau_loc: Noise (tau) prior location.
        tau_scale: Noise (tau) prior scale.
        epsilon_min: Lower bound of the epsilon prior.
        sigma_loc: Fixed sigma prior location, or None for data-dependent.
        sigma_scale: Fixed sigma prior scale, or None for data-dependent.
        sigma_low_fraction: Headroom fraction targeted by the 20th percentile.
        sigma_high_fraction: Headroom fraction targeted by the 80th percentile.
        sigma_calibration_samples: Monte Carlo samples per grid cell.
        sigma_calibration_seed: Seed making the calibration deterministic.
    """

    lengthscale_loc: float = -1.23
    lengthscale_scale: float = 2.12
    tau_loc: float = 0.01
    tau_scale: float = 0.01
    epsilon_min: float = 0.0
    sigma_loc: float | None = None
    sigma_scale: float | None = None
    sigma_low_fraction: float = 0.5
    sigma_high_fraction: float = 0.75
    sigma_calibration_samples: int = 1000
    sigma_calibration_seed: int = 0

    def __post_init__(self) -> None:
        """Validates that the fixed sigma prior is fully specified or absent."""
        if (self.sigma_loc is None) != (self.sigma_scale is None):
            raise ValueError("Set both sigma_loc and sigma_scale, or neither.")


@dataclass(frozen=True)
class APExGPHyperparameters:
    """Constrained values of all fitted quantities.

    Attributes:
        epsilon: Saturation gap of the mean function.
        theta1: First mean-function parameter.
        theta2: Second mean-function parameter.
        sigma: Kernel output scale (sqrt of gpytorch's outputscale).
        lengthscale: Kernel lengthscale (log10-size units).
        tau: Observation noise standard deviation (>= 0.01).
    """

    epsilon: float
    theta1: float
    theta2: float
    sigma: float
    lengthscale: float
    tau: float


def epsilon_upper_width(max_auroc: float, epsilon_min: float) -> float:
    """Width of the epsilon prior support, (1 - max_auroc) - epsilon_min.

    Args:
        max_auroc: Largest observed pilot AUROC, y'.
        epsilon_min: Lower bound of the epsilon prior.

    Returns:
        The support width, floored at 1e-6 so the prior stays proper.
    """
    return max(1.0 - max_auroc - epsilon_min, 1e-6)


def initial_hyperparameters(
    sizes: np.ndarray,
    aurocs: np.ndarray,
    mean_function: MeanFunctionName,
    priors: APExGPPriors,
    sigma: float = 0.01,
) -> APExGPHyperparameters:
    """Data-driven starting point, used for restart 1 of the MAP optimization.

    Epsilon starts at the midpoint of its prior support. Theta comes from a
    linear least-squares fit in a linearizing transform of the mean function
    (log-log for the power law, tan for arctan), clipped to the constraints.
    Sigma starts at the given value (``fit`` passes the sigma prior's loc), tau
    just above its 0.01 floor, and the lengthscale at gpytorch's default.

    Args:
        sizes: Pilot sizes, shape (R,).
        aurocs: Pilot AUROCs, shape (R,).
        mean_function: "power_law" or "arctan".
        priors: Prior configuration (for epsilon_min).
        sigma: Initial kernel output scale (> 0).

    Returns:
        Initial hyperparameters satisfying all constraints.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    aurocs = np.asarray(aurocs, dtype=np.float64)
    width = epsilon_upper_width(float(aurocs.max()), priors.epsilon_min)
    epsilon = priors.epsilon_min + 0.5 * width

    if mean_function == "power_law":
        log_x = np.log(sizes)
        log_gap = np.log(np.clip((1.0 - epsilon) - aurocs, 1e-6, None))
        slope = np.polyfit(log_x, log_gap, 1)[0] if len(sizes) > 1 else -0.5
        theta2 = float(np.clip(slope, -0.99, -0.01))
        theta1 = float(np.exp(np.mean(log_gap - theta2 * log_x)))
    else:
        scaled_x = sizes * (math.pi / 2.0)
        target = np.tan(np.clip(aurocs + epsilon, 1e-6, 1.0 - 1e-6) * math.pi / 2)
        slope = np.polyfit(scaled_x, target, 1)[0] if len(sizes) > 1 else 0.0
        theta1 = float(max(slope, 1e-6))
        theta2 = float(max(np.mean(target - theta1 * scaled_x), 0.0))

    return APExGPHyperparameters(
        epsilon=epsilon,
        theta1=theta1,
        theta2=theta2,
        sigma=max(sigma, 1e-4),
        lengthscale=math.log(2.0),  # gpytorch default, softplus(0)
        tau=0.0125,
    )


def _softplus_inverse(value: float) -> float:
    """Inverse of softplus for value > 0, in a numerically stable form."""
    value = max(value, 1e-9)
    return value + math.log(-math.expm1(-value))


def _logit(p: float) -> float:
    """Logit of p, with p clipped to (1e-6, 1 - 1e-6)."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class APExGPRegressor:
    """MAP-fitted GP over learning curves (the APEx-GP core, no PyHealth needed).

    All six quantities are optimized jointly with Adam on an unconstrained
    float64 vector ``raw``, mirroring the gpytorch constraints in the authors'
    code:

    - epsilon = epsilon_min + width * sigmoid(raw[0])  (Interval)
    - theta1 = softplus(raw[1])  (Positive)
    - theta2 = -1 + sigmoid(raw[2]) (power law, Interval(-1, 0)) or
      softplus(raw[2]) (arctan, Positive)
    - sigma**2 = softplus(raw[3])  (outputscale, Positive)
    - lengthscale = softplus(raw[4]) + 1e-6  (Positive; tiny floor added)
    - tau**2 = 1e-4 + softplus(raw[5])  (noise, GreaterThan(1e-4))

    Priors are evaluated on sigma, lengthscale and tau. The objective is
    log p(y | x) + log p(eta) in the constrained space (no Jacobian term).

    Restart 0 starts from ``raw = 0`` (the authors' initialization), restart 1
    from ``initial_hyperparameters`` (sigma at the sigma prior's loc), and any
    later restarts from ``raw = 0`` plus Gaussian noise. The restart with the
    highest log posterior is kept. With the default lr=0.01, restart 0 needs
    up to ~16-20k iterations; restart 1 typically stops early after ~2-4k.

    Args:
        mean_function: "power_law" (default) or "arctan".
        priors: Prior configuration. Defaults to ``APExGPPriors()``.
        lr: Adam learning rate (authors: 0.01).
        n_iterations: Maximum Adam iterations per restart.
        n_restarts: Number of restarts (see above).
        restart_scale: Std of the noise added to ``raw`` for restarts >= 2.
        tol: Minimum log-posterior gain that counts as progress.
        patience: Stop a restart after this many iterations without progress.
        jitter: Initial Cholesky jitter.
        seed: Seed for restart noise.

    Attributes:
        hyperparameters_: Fitted ``APExGPHyperparameters`` (after ``fit``).
        log_posterior_: Unnormalized log posterior at the optimum.
        sigma_prior_loc_: Location of the sigma prior used in the fit.
        sigma_prior_scale_: Scale of the sigma prior used in the fit.
        n_iterations_run_: Adam iterations used by each restart.
    """

    def __init__(
        self,
        mean_function: MeanFunctionName = "power_law",
        priors: APExGPPriors | None = None,
        lr: float = 0.01,
        n_iterations: int = DEFAULT_N_ITERATIONS,
        n_restarts: int = 2,
        restart_scale: float = 1.0,
        tol: float = 1e-6,
        patience: int = 500,
        jitter: float = 1e-8,
        seed: int = 0,
    ) -> None:
        if mean_function not in MEAN_FUNCTIONS:
            raise ValueError(
                f"mean_function must be one of {sorted(MEAN_FUNCTIONS)}; "
                f"got {mean_function!r}."
            )
        if n_restarts < 1 or n_iterations < 1 or patience < 1:
            raise ValueError("n_restarts, n_iterations and patience must be >= 1.")
        self.mean_function = mean_function
        self.priors = priors if priors is not None else APExGPPriors()
        self.lr = lr
        self.n_iterations = n_iterations
        self.n_restarts = n_restarts
        self.restart_scale = restart_scale
        self.tol = tol
        self.patience = patience
        self.jitter = jitter
        self.seed = seed
        self.hyperparameters_: APExGPHyperparameters | None = None
        self.log_posterior_: float | None = None
        self.sigma_prior_loc_: float | None = None
        self.sigma_prior_scale_: float | None = None
        self.n_iterations_run_: list[int] = []
        self._x_train: torch.Tensor | None = None
        self._y_train: torch.Tensor | None = None
        self._epsilon_width: float = 1.0

    # ----------------------------------------------------------------- params
    def _constrain(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Maps the unconstrained vector to constrained parameters.

        Args:
            raw: Unconstrained parameters, shape (6,).

        Returns:
            Dict of scalar tensors keyed by ``_PARAM_NAMES``.
        """
        if self.mean_function == "power_law":
            theta2 = -1.0 + torch.sigmoid(raw[2])
        else:
            theta2 = F.softplus(raw[2])
        epsilon = self.priors.epsilon_min + self._epsilon_width * torch.sigmoid(raw[0])
        return {
            "epsilon": epsilon,
            "theta1": F.softplus(raw[1]),
            "theta2": theta2,
            "sigma": torch.sqrt(F.softplus(raw[3])),
            "lengthscale": F.softplus(raw[4]) + _LENGTHSCALE_FLOOR,
            "tau": torch.sqrt(NOISE_VARIANCE_FLOOR + F.softplus(raw[5])),
        }

    def _unconstrain(self, hp: APExGPHyperparameters) -> torch.Tensor:
        """Inverse of ``_constrain``.

        Args:
            hp: Constrained hyperparameters (tau must exceed 0.01).

        Returns:
            Unconstrained float64 vector, shape (6,).
        """
        if self.mean_function == "power_law":
            raw_theta2 = _logit(hp.theta2 + 1.0)
        else:
            raw_theta2 = _softplus_inverse(hp.theta2)
        eps_frac = (hp.epsilon - self.priors.epsilon_min) / self._epsilon_width
        values = [
            _logit(eps_frac),
            _softplus_inverse(hp.theta1),
            raw_theta2,
            _softplus_inverse(hp.sigma**2),
            _softplus_inverse(hp.lengthscale - _LENGTHSCALE_FLOOR),
            _softplus_inverse(hp.tau**2 - NOISE_VARIANCE_FLOOR),
        ]
        return torch.tensor(values, dtype=torch.float64)

    def _set_sigma_prior(self, max_auroc: float) -> None:
        """Fixes the sigma prior for this fit (data-dependent unless overridden).

        Args:
            max_auroc: Largest observed pilot AUROC.
        """
        pr = self.priors
        if pr.sigma_loc is not None and pr.sigma_scale is not None:
            self.sigma_prior_loc_, self.sigma_prior_scale_ = (
                pr.sigma_loc,
                pr.sigma_scale,
            )
            return
        headroom = (1.0 - pr.epsilon_min) - max_auroc
        self.sigma_prior_loc_, self.sigma_prior_scale_ = calibrate_sigma_prior(
            pr.tau_loc,
            pr.tau_scale,
            desired_low=pr.sigma_low_fraction * headroom,
            desired_high=pr.sigma_high_fraction * headroom,
            num_samples=pr.sigma_calibration_samples,
            seed=pr.sigma_calibration_seed,
        )

    # -------------------------------------------------------------- objective
    def log_prior(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Log prior log p(eta) of the constrained parameters.

        Args:
            params: Output of ``_constrain``.

        Returns:
            Scalar log prior (theta contributes nothing).

        Raises:
            RuntimeError: If the sigma prior has not been set by ``fit``.
        """
        if self.sigma_prior_loc_ is None or self.sigma_prior_scale_ is None:
            raise RuntimeError("Sigma prior not set; call fit().")
        pr = self.priors
        return (
            truncated_normal_log_prob(
                params["lengthscale"], pr.lengthscale_loc, pr.lengthscale_scale
            )
            + truncated_normal_log_prob(
                params["sigma"], self.sigma_prior_loc_, self.sigma_prior_scale_
            )
            + truncated_normal_log_prob(params["tau"], pr.tau_loc, pr.tau_scale)
            - math.log(self._epsilon_width)  # Uniform density on its support
        )

    def log_posterior(
        self, raw: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Unnormalized log posterior log p(y | x) + log p(eta).

        Args:
            raw: Unconstrained parameters, shape (6,).
            x: Pilot sizes, shape (R,).
            y: Pilot AUROCs, shape (R,).

        Returns:
            Scalar objective to maximize.
        """
        p = self._constrain(raw)
        mean = MEAN_FUNCTIONS[self.mean_function](
            x, p["epsilon"], p["theta1"], p["theta2"]
        )
        kernel = log_rbf_kernel(x, x, p["sigma"], p["lengthscale"])
        lml = gp_log_marginal_likelihood(y, mean, kernel, p["tau"], self.jitter)
        return lml + self.log_prior(p)

    def _optimize(
        self, start: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, float, int]:
        """Runs Adam from ``start`` and returns the best iterate.

        Stops early once ``patience`` consecutive iterations improve the best
        log posterior by no more than ``tol``.

        Args:
            start: Initial unconstrained vector, shape (6,).
            x: Pilot sizes.
            y: Pilot AUROCs.

        Returns:
            A tuple ``(best_raw, best_objective, iterations_run)``; the
            objective is -inf if every evaluation failed.
        """
        raw = nn.Parameter(start.clone())
        optimizer = torch.optim.Adam([raw], lr=self.lr)
        best_raw, best_obj = start.clone(), -math.inf
        stale = 0
        iterations = 0
        for _ in range(self.n_iterations + 1):
            iterations += 1
            optimizer.zero_grad()
            try:
                objective = self.log_posterior(raw, x, y)
            except RuntimeError:  # Cholesky failure
                break
            if not torch.isfinite(objective):
                break
            value = objective.item()
            gain = value - best_obj
            if gain > 0:
                best_obj, best_raw = value, raw.detach().clone()
            stale = 0 if gain > self.tol else stale + 1
            if stale >= self.patience:
                break
            (-objective).backward()
            optimizer.step()
        return best_raw, best_obj, iterations

    # ------------------------------------------------------------- public API
    def fit(
        self, sizes: Sequence[float] | np.ndarray, aurocs: Sequence[float] | np.ndarray
    ) -> APExGPRegressor:
        """MAP-fits the GP to pilot (size, AUROC) pairs.

        Args:
            sizes: Positive pilot training-set sizes, shape (R,), R >= 2.
            aurocs: Mean test AUROC at each size, values in [0, 1].

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If inputs have the wrong shape or range.
            RuntimeError: If every restart fails.
        """
        x_np = np.asarray(sizes, dtype=np.float64).reshape(-1)
        y_np = np.asarray(aurocs, dtype=np.float64).reshape(-1)
        if x_np.shape != y_np.shape or len(x_np) < 2:
            raise ValueError("sizes and aurocs must be 1-D, equal length, R >= 2.")
        if np.any(x_np <= 0) or np.any((y_np < 0) | (y_np > 1)):
            raise ValueError("sizes must be > 0 and aurocs within [0, 1].")

        max_auroc = float(y_np.max())
        self._epsilon_width = epsilon_upper_width(max_auroc, self.priors.epsilon_min)
        self._set_sigma_prior(max_auroc)
        x = torch.as_tensor(x_np, dtype=torch.float64)
        y = torch.as_tensor(y_np, dtype=torch.float64)

        generator = torch.Generator().manual_seed(self.seed)
        best_raw, best_obj = None, -math.inf
        self.n_iterations_run_ = []
        for restart in range(self.n_restarts):
            if restart == 1:
                init = initial_hyperparameters(
                    x_np,
                    y_np,
                    self.mean_function,
                    self.priors,
                    sigma=self.sigma_prior_loc_,
                )
                start = self._unconstrain(init)
            else:
                start = torch.zeros(6, dtype=torch.float64)  # authors' init
                if restart >= 2:
                    noise = torch.randn(6, generator=generator, dtype=torch.float64)
                    start = start + self.restart_scale * noise
            raw, obj, n_run = self._optimize(start, x, y)
            self.n_iterations_run_.append(n_run)
            logger.debug(
                "APEx-GP restart %d: log posterior %.4f after %d iterations",
                restart,
                obj,
                n_run,
            )
            if obj > best_obj:
                best_raw, best_obj = raw, obj
        if best_raw is None:
            raise RuntimeError("All APEx-GP restarts failed.")

        with torch.no_grad():
            params = self._constrain(best_raw)
        self.hyperparameters_ = APExGPHyperparameters(
            **{name: float(params[name]) for name in _PARAM_NAMES}
        )
        self.log_posterior_ = best_obj
        self._x_train, self._y_train = x, y
        return self

    def _check_fitted(self) -> APExGPHyperparameters:
        """Returns fitted hyperparameters or raises if ``fit`` was not called."""
        if self.hyperparameters_ is None or self._x_train is None:
            raise RuntimeError("APExGPRegressor is not fitted; call fit() first.")
        return self.hyperparameters_

    def predict(
        self, sizes: Sequence[float] | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Untruncated Gaussian posterior predictive at new sizes.

        Args:
            sizes: Positive query sizes, shape (Q,).

        Returns:
            A tuple ``(mean, cov)`` of float64 tensors, shapes (Q,), (Q, Q).
        """
        hp = self._check_fitted()
        x_test = torch.as_tensor(
            np.asarray(sizes, dtype=np.float64).reshape(-1), dtype=torch.float64
        )
        mean_fn = MEAN_FUNCTIONS[self.mean_function]

        def prior_mean(x: torch.Tensor) -> torch.Tensor:
            return mean_fn(x, hp.epsilon, hp.theta1, hp.theta2)

        with torch.no_grad():
            return gp_posterior_predictive(
                self._x_train,
                self._y_train,
                x_test,
                prior_mean,
                hp.sigma,
                hp.lengthscale,
                hp.tau,
                self.jitter,
            )

    def predictive_marginals(
        self, sizes: Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Location and scale of each untruncated predictive marginal.

        Args:
            sizes: Positive query sizes, shape (Q,).

        Returns:
            A tuple ``(loc, scale)`` of NumPy arrays, shape (Q,).
        """
        mean, cov = self.predict(sizes)
        scale = torch.sqrt(torch.clamp(torch.diagonal(cov), min=1e-12))
        return mean.cpu().numpy(), scale.cpu().numpy()

    def predict_interval(
        self, sizes: Sequence[float] | np.ndarray, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean and highest-density interval of the [0, 1]-truncated predictive.

        Args:
            sizes: Positive query sizes, shape (Q,).
            level: Interval probability mass in (0, 1).

        Returns:
            A tuple ``(mean, lower, upper)`` of NumPy arrays, shape (Q,).
        """
        loc, scale = self.predictive_marginals(sizes)
        return truncated_normal_summary(loc, scale, level)

    def evaluate(
        self,
        sizes: Sequence[float] | np.ndarray,
        observed: Sequence[float] | np.ndarray,
        delta: float = 0.01,
        level: float = 0.95,
    ) -> dict[str, float]:
        """Paper metrics against observed AUROCs at larger sizes.

        Args:
            sizes: Query sizes, shape (Q,).
            observed: Observed AUROC at each size, shape (Q,).
            delta: Half-width for the quantized likelihood.
            level: HDI level for coverage.

        Returns:
            Dict with "rmse", "quantized_likelihood" (mean over sizes) and
            "coverage".
        """
        observed = np.asarray(observed, dtype=np.float64).reshape(-1)
        loc, scale = self.predictive_marginals(sizes)
        mean, lower, upper = truncated_normal_summary(loc, scale, level)
        return {
            "rmse": rmse(observed, mean),
            "quantized_likelihood": float(
                np.mean(quantized_likelihood(observed, loc, scale, delta))
            ),
            "coverage": interval_coverage(observed, lower, upper),
        }


# =========================================================================== #
# 2. Evaluation on the [0, 1]-truncated predictive (NumPy/SciPy)
# =========================================================================== #
def _truncnorm(loc: float, scale: float, low: float = 0.0, high: float = 1.0) -> Any:
    """Frozen SciPy normal(loc, scale) truncated to [low, high]."""
    return stats.truncnorm(
        (low - loc) / scale, (high - loc) / scale, loc=loc, scale=scale
    )


def truncated_normal_hdi(
    loc: float, scale: float, level: float = 0.95, low: float = 0.0, high: float = 1.0
) -> tuple[float, float]:
    """Highest-density interval of a normal truncated to [low, high].

    The truncated density is symmetric and decreasing in |x - loc| on its
    support, so the HDI is [loc - c, loc + c] intersected with [low, high],
    with c found by root finding so that the interval holds ``level`` mass.

    Args:
        loc: Untruncated mean.
        scale: Untruncated standard deviation (> 0).
        level: Probability mass in (0, 1).
        low: Lower truncation bound.
        high: Upper truncation bound.

    Returns:
        The interval ``(lower, upper)``.

    Raises:
        ValueError: If ``level`` is not in (0, 1).
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}.")
    dist = _truncnorm(loc, scale, low, high)

    def excess_mass(c: float) -> float:
        return float(dist.cdf(min(loc + c, high)) - dist.cdf(max(loc - c, low))) - level

    c_max = max(abs(loc - low), abs(high - loc))
    c = brentq(excess_mass, 0.0, c_max, xtol=1e-10)
    return max(loc - c, low), min(loc + c, high)


def truncated_normal_summary(
    loc: np.ndarray, scale: np.ndarray, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and HDI of each [0, 1]-truncated normal marginal.

    Args:
        loc: Untruncated means, shape (Q,).
        scale: Untruncated standard deviations, shape (Q,).
        level: HDI probability mass.

    Returns:
        A tuple ``(mean, lower, upper)``, each shape (Q,).
    """
    loc = np.asarray(loc, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    mean = np.array([_truncnorm(m, s).mean() for m, s in zip(loc, scale, strict=True)])
    bounds = np.array(
        [truncated_normal_hdi(m, s, level) for m, s in zip(loc, scale, strict=True)]
    )
    return mean, bounds[:, 0], bounds[:, 1]


def quantized_likelihood(
    observed: np.ndarray, loc: np.ndarray, scale: np.ndarray, delta: float = 0.01
) -> np.ndarray:
    """Probability the truncated predictive assigns to (y* - delta, y* + delta).

    Args:
        observed: Observed AUROCs, shape (Q,).
        loc: Untruncated predictive means, shape (Q,).
        scale: Untruncated predictive standard deviations, shape (Q,).
        delta: Half-width of the window.

    Returns:
        Per-size probabilities, shape (Q,).
    """
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    loc = np.asarray(loc, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    probs = []
    for y, m, s in zip(observed, loc, scale, strict=True):
        dist = _truncnorm(m, s)
        probs.append(dist.cdf(y + delta) - dist.cdf(y - delta))
    return np.asarray(probs, dtype=np.float64)


def interval_coverage(
    observed: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fraction of observed values inside [lower, upper].

    Args:
        observed: Observed AUROCs, shape (Q,).
        lower: Interval lower bounds, shape (Q,).
        upper: Interval upper bounds, shape (Q,).

    Returns:
        Coverage in [0, 1].
    """
    observed = np.asarray(observed, dtype=np.float64)
    return float(np.mean((observed >= lower) & (observed <= upper)))


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Root-mean-squared error.

    Args:
        observed: Observed values, shape (Q,).
        predicted: Predicted values, shape (Q,).

    Returns:
        The RMSE.
    """
    diff = np.asarray(observed, np.float64) - np.asarray(predicted, np.float64)
    return float(np.sqrt(np.mean(diff**2)))


# =========================================================================== #
# 3a. Pilot experiment (scikit-learn)
# =========================================================================== #
@dataclass
class PilotResults:
    """Outcome of the pilot experiment.

    Attributes:
        sizes: Pilot sizes, shape (R,).
        auroc_mean: Test AUROC averaged over seeds, shape (R,).
        auroc_per_seed: Test AUROC per seed and size, shape (n_seeds, R).
        best_c_per_seed: Selected C per seed and size, shape (n_seeds, R).
        classifier: Pipeline trained on the largest subset of the first seed.
    """

    sizes: np.ndarray
    auroc_mean: np.ndarray
    auroc_per_seed: np.ndarray
    best_c_per_seed: np.ndarray
    classifier: Pipeline


def split_by_group(
    groups: np.ndarray,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits sample indices into group-disjoint train/val/test sets.

    Fractions apply to the number of groups (subjects/patients). Validation and
    test counts are rounded (minimum 1 each); training gets the rest.

    Args:
        groups: Group identifier per sample, shape (N,).
        fractions: Train, validation and test fractions summing to 1.
        seed: Seed for shuffling groups.

    Returns:
        Sorted index arrays ``(train, val, test)``.

    Raises:
        ValueError: If fractions are invalid or there are too few groups.
    """
    if min(fractions) < 0 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be non-negative and sum to 1.")
    groups = np.asarray(groups)
    unique = np.unique(groups)
    n = len(unique)
    n_val = max(1, int(round(fractions[1] * n)))
    n_test = max(1, int(round(fractions[2] * n)))
    if n - n_val - n_test < 1:
        raise ValueError(f"{n} groups are too few for a train/val/test split.")
    order = np.random.default_rng(seed).permutation(unique)
    test_g, val_g, train_g = (
        order[:n_test],
        order[n_test : n_test + n_val],
        order[n_test + n_val :],
    )
    return tuple(  # type: ignore[return-value]
        np.flatnonzero(np.isin(groups, g)) for g in (train_g, val_g, test_g)
    )


def draw_nested_subsets(
    labels: np.ndarray,
    train_indices: np.ndarray,
    sizes: Sequence[int],
    seed: int = 0,
    max_attempts: int = 1000,
) -> list[np.ndarray]:
    """Draws nested subsets (prefixes of one random order) containing both classes.

    Args:
        labels: Binary labels for all samples.
        train_indices: Indices of the training split.
        sizes: Strictly increasing subset sizes.
        seed: Seed for the random order.
        max_attempts: Redraws allowed until the smallest subset has both classes.

    Returns:
        One index array per size, each containing all smaller ones.

    Raises:
        ValueError: If sizes are invalid or no valid order is found.
    """
    sizes = [int(s) for s in sizes]
    if any(b <= a for a, b in zip(sizes, sizes[1:], strict=False)) or sizes[0] < 2:
        raise ValueError(f"sizes must be strictly increasing and >= 2; got {sizes}.")
    if sizes[-1] > len(train_indices):
        raise ValueError(
            f"Largest subset ({sizes[-1]}) exceeds the training split "
            f"({len(train_indices)} samples)."
        )
    train_labels = np.asarray(labels)[train_indices]
    if len(np.unique(train_labels)) < 2:
        raise ValueError("Training split contains a single class.")
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        order = rng.permutation(len(train_indices))
        if len(np.unique(train_labels[order[: sizes[0]]])) == 2:
            ordered = np.asarray(train_indices)[order]
            return [ordered[:size].copy() for size in sizes]
    raise ValueError("Could not draw a smallest subset containing both classes.")


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    seed: int = 0,
) -> tuple[Pipeline, float]:
    """Trains standardized L2 logistic regression, choosing C by validation AUROC.

    Ties keep the earliest C in ``c_grid``. If the validation split has a
    single class, C = 1.0 is used and a warning is logged.

    Args:
        x_train: Training features, shape (n, d).
        y_train: Training labels, shape (n,).
        x_val: Validation features.
        y_val: Validation labels.
        c_grid: Candidate inverse regularization strengths.
        seed: Random state passed to the solver.

    Returns:
        A tuple ``(pipeline, best_c)``.
    """

    def make(c: float) -> Pipeline:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, random_state=seed),
        )
        return model.fit(x_train, y_train)

    if len(np.unique(y_val)) < 2:
        logger.warning("Validation split has one class; using C=1.0.")
        return make(1.0), 1.0
    best_model, best_c, best_score = None, float(c_grid[0]), -math.inf
    for c in c_grid:
        model = make(float(c))
        score = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
        if score > best_score:
            best_model, best_c, best_score = model, float(c), score
    return best_model, best_c  # type: ignore[return-value]


def run_pilot_experiment(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sizes: Sequence[int] = DEFAULT_SUBSET_SIZES,
    n_seeds: int = 5,
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    seed: int = 0,
) -> PilotResults:
    """Measures test AUROC of logistic regression on nested pilot subsets.

    For each seed: a fresh group-disjoint split, fresh nested subsets, one
    classifier per size (C tuned on validation), test AUROC recorded.

    Args:
        features: Feature matrix, shape (N, d).
        labels: Binary labels, shape (N,).
        groups: Group (subject/patient) ID per sample, shape (N,).
        sizes: Strictly increasing pilot sizes.
        n_seeds: Number of random splits/subset draws to average over.
        split_fractions: Train/val/test fractions of groups.
        c_grid: Candidate C values.
        seed: Base seed; seed ``s`` uses ``seed + s``.

    Returns:
        A ``PilotResults``.

    Raises:
        ValueError: If a test split contains a single class.
    """
    labels = np.asarray(labels).astype(np.int64)
    n_sizes = len(sizes)
    aurocs = np.empty((n_seeds, n_sizes))
    best_cs = np.empty((n_seeds, n_sizes))
    classifier: Pipeline | None = None
    for s in range(n_seeds):
        run_seed = seed + s
        train, val, test = split_by_group(groups, split_fractions, run_seed)
        if len(np.unique(labels[test])) < 2:
            raise ValueError(f"Seed {run_seed}: test split has a single class.")
        subsets = draw_nested_subsets(labels, train, sizes, run_seed)
        for r, idx in enumerate(subsets):
            model, best_c = fit_logistic_regression(
                features[idx], labels[idx], features[val], labels[val], c_grid, run_seed
            )
            scores = model.predict_proba(features[test])[:, 1]
            aurocs[s, r] = roc_auc_score(labels[test], scores)
            best_cs[s, r] = best_c
            if s == 0 and r == n_sizes - 1:
                classifier = model
        logger.info("Pilot seed %d AUROCs: %s", run_seed, np.round(aurocs[s], 4))
    return PilotResults(
        sizes=np.asarray(sizes, dtype=np.float64),
        auroc_mean=aurocs.mean(axis=0),
        auroc_per_seed=aurocs,
        best_c_per_seed=best_cs,
        classifier=classifier,  # type: ignore[arg-type]
    )


# =========================================================================== #
# 3b. PyHealth model wrapper
# =========================================================================== #
def _to_numpy(value: Any) -> np.ndarray:
    """Converts a tensor, array or scalar to a NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class APExGP(BaseModel):
    """APEx-GP as a PyHealth model.

    ``fit()`` runs the pilot experiment on this model's dataset (group-disjoint
    splits, nested subsets, validation-tuned logistic regression, test AUROC
    averaged over seeds) and MAP-fits the GP to the resulting learning curve.
    ``predict_interval()`` forecasts AUROC at larger sizes. ``forward()`` scores
    batches with the logistic regression trained on the largest pilot subset,
    so ``Trainer.evaluate`` and inference work.

    The authors train a torch logistic regression with weight decay (L-BFGS) on
    frozen image embeddings without standardization; the scikit-learn L2
    logistic regression with ``StandardScaler`` used here is an intentional
    equivalent for tabular features such as EEG band powers.

    Note:
        The model has no trainable torch parameters; train it with ``fit()``,
        not ``Trainer.train()``. The fitted GP is a Python attribute and is not
        part of ``state_dict()``; only the classifier weights are.

    Args:
        dataset: A ``SampleDataset`` with exactly one input feature (a vector,
            e.g. ``"tensor"`` processor) and exactly one binary label. Each
            sample must also carry ``group_key`` (default ``"patient_id"``).
        subset_sizes: Strictly increasing pilot sizes in samples.
        n_seeds: Number of random splits/subset draws averaged per size.
        split_fractions: Train/val/test fractions of groups.
        c_grid: Candidate logistic regression C values.
        group_key: Sample key holding the subject/patient ID used for splits.
        mean_function: GP mean, "power_law" (default) or "arctan".
        priors: GP priors. Defaults to ``APExGPPriors()``.
        lr: Adam learning rate for the GP MAP fit (authors: 0.01).
        n_iterations: Maximum Adam iterations per restart.
        n_restarts: Number of optimizer restarts (restart 0 uses the authors'
            initialization; see ``APExGPRegressor``).
        seed: Base random seed.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": f"S{i // 60:03d}", "features": x.tolist(),
        ...      "label": int(t)}
        ...     for i, (x, t) in enumerate(zip(X, y))
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = APExGP(dataset).fit()
        >>> mean, lower, upper = model.predict_interval([720, 1440, 5000])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        subset_sizes: Sequence[int] = DEFAULT_SUBSET_SIZES,
        n_seeds: int = 5,
        split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
        c_grid: Sequence[float] = DEFAULT_C_GRID,
        group_key: str = "patient_id",
        mean_function: MeanFunctionName = "power_law",
        priors: APExGPPriors | None = None,
        lr: float = 0.01,
        n_iterations: int = DEFAULT_N_ITERATIONS,
        n_restarts: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset)
        if len(self.feature_keys) != 1 or len(self.label_keys) != 1:
            raise ValueError(
                "APExGP needs exactly one input feature and one label; got "
                f"{self.feature_keys} and {self.label_keys}."
            )
        if self.mode != "binary":
            raise ValueError(f"APExGP needs a binary label; got mode={self.mode!r}.")
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.subset_sizes = tuple(int(s) for s in subset_sizes)
        self.n_seeds = n_seeds
        self.split_fractions = split_fractions
        self.c_grid = tuple(c_grid)
        self.group_key = group_key
        self.seed = seed
        self.gp = APExGPRegressor(
            mean_function=mean_function,
            priors=priors,
            lr=lr,
            n_iterations=n_iterations,
            n_restarts=n_restarts,
            seed=seed,
        )
        self.pilot_results_: PilotResults | None = None

        n_features = int(_to_numpy(next(iter(dataset))[self.feature_key]).size)
        self.register_buffer("clf_weight", torch.zeros(n_features))
        self.register_buffer("clf_bias", torch.zeros(()))
        self.register_buffer("is_fitted", torch.tensor(False))

    # ------------------------------------------------------------- fit steps
    def _dataset_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collects features, binary labels and group IDs from the dataset.

        Returns:
            ``(features (N, d), labels (N,), groups (N,))``.

        Raises:
            KeyError: If a sample lacks ``group_key``.
        """
        features, labels, groups = [], [], []
        for sample in self.dataset:
            if self.group_key not in sample:
                raise KeyError(
                    f"Sample has no {self.group_key!r}; set group_key to the "
                    "subject/patient ID field."
                )
            features.append(_to_numpy(sample[self.feature_key]).reshape(-1))
            labels.append(int(round(float(_to_numpy(sample[self.label_key]).flat[0]))))
            groups.append(str(sample[self.group_key]))
        return (
            np.vstack(features).astype(np.float64),
            np.asarray(labels, dtype=np.int64),
            np.asarray(groups),
        )

    def _load_classifier(self, classifier: Pipeline) -> None:
        """Folds a StandardScaler + LogisticRegression pipeline into buffers.

        logit = ((x - mu) / s) @ w + b = x @ (w / s) + (b - (mu / s) @ w).

        Args:
            classifier: Fitted pipeline from ``fit_logistic_regression``.
        """
        scaler: StandardScaler = classifier[0]
        logreg: LogisticRegression = classifier[-1]
        coef = logreg.coef_.reshape(-1) / scaler.scale_
        bias = float(logreg.intercept_[0]) - float(np.dot(scaler.mean_, coef))
        self.clf_weight.copy_(torch.as_tensor(coef, dtype=self.clf_weight.dtype))
        self.clf_bias.fill_(bias)
        self.is_fitted.fill_(True)

    def fit(self) -> APExGP:
        """Runs the pilot experiment and MAP-fits the GP.

        Returns:
            ``self``, fitted.
        """
        features, labels, groups = self._dataset_arrays()
        self.pilot_results_ = run_pilot_experiment(
            features,
            labels,
            groups,
            sizes=self.subset_sizes,
            n_seeds=self.n_seeds,
            split_fractions=self.split_fractions,
            c_grid=self.c_grid,
            seed=self.seed,
        )
        self._load_classifier(self.pilot_results_.classifier)
        self.gp.fit(self.pilot_results_.sizes, self.pilot_results_.auroc_mean)
        logger.info("APEx-GP hyperparameters: %s", self.gp.hyperparameters_)
        return self

    # ------------------------------------------------------------ prediction
    def predict_interval(
        self, sizes: Sequence[float] | np.ndarray, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecasts AUROC at larger training-set sizes.

        Args:
            sizes: Query sizes, shape (Q,).
            level: Highest-density interval mass.

        Returns:
            ``(mean, lower, upper)`` of the [0, 1]-truncated predictive.
        """
        return self.gp.predict_interval(sizes, level)

    def forward(
        self, **kwargs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """Scores a batch with the classifier trained on the largest pilot subset.

        Args:
            **kwargs: Batch from a PyHealth dataloader; must contain the feature
                key and may contain the label key.

        Returns:
            Dict with "logit" and "y_prob" of shape (batch, 1), plus "loss" and
            "y_true" when the label is present.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if not bool(self.is_fitted):
            raise RuntimeError("APExGP is not fitted; call fit() first.")
        x = kwargs[self.feature_key]
        if isinstance(x, tuple):
            x = x[0]
        x = x.to(self.device, dtype=self.clf_weight.dtype).reshape(x.shape[0], -1)
        logit = (x @ self.clf_weight + self.clf_bias).unsqueeze(-1)
        results = {
            "logit": logit,
            "y_prob": self.label_processor.activation(logit),
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key]
            if isinstance(y_true, tuple):
                y_true = y_true[0]
            y_true = y_true.to(self.device, dtype=logit.dtype).reshape(logit.shape)
            results["loss"] = self.label_processor.loss(logit, y_true)
            results["y_true"] = y_true
        return results
