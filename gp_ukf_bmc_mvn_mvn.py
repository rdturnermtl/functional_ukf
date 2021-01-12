# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: gp_ukf
#     language: python
#     name: gp_ukf
# ---

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from gp_ukf_core import gp_ukf
from scipy.stats import multivariate_normal, mvn, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

np.random.seed(0)

# +
# For 3D try: d = 3, n_grid = 15
d = 2
n_grid = 50
n_eval_grid = 8

lb = 1.5
ub = 3.5

# +
# Build grids
xgrid = np.linspace(lb, ub, n_grid + 1)
dx = np.median(np.diff(xgrid))
xgrid = np.array(list(product(xgrid[:-1], repeat=d)))

eval_grid = np.linspace(lb, ub, n_eval_grid + 1)
eval_grid = np.array(list(product(eval_grid[:-1], repeat=d)))
# -

print(f"Ground truth grid of shape: {xgrid.shape}")
print(f"Eval grid of shape: {eval_grid.shape}")

# Random Gaussian example
mvn_mu = np.random.randn(d)
mvn_cov = np.random.randn(d, d)
mvn_cov = np.dot(mvn_cov, mvn_cov.T)

# The exact answer using same grid method (real func value, not from GP)
prob_full = np.sum(np.exp(multivariate_normal.logpdf(xgrid, mvn_mu, mvn_cov))) * (dx ** d)


def warp_func(loglik_surface):
    n_pts, n_grid = loglik_surface.shape
    assert xgrid.shape == (n_grid, d)

    lik_surface = np.exp(loglik_surface)
    int_val = np.sum(lik_surface, axis=1, keepdims=True) * (dx ** d)
    assert int_val.shape == (n_pts, 1)
    return int_val


prob_full_ = warp_func(multivariate_normal.logpdf(xgrid, mvn_mu, mvn_cov)[None, :]).item()
assert np.isclose(prob_full_, prob_full)

# Get the evals for BMC
logpdf = multivariate_normal.logpdf(eval_grid, mvn_mu, mvn_cov)

# Setup and train GP to the observations on the nrg
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(0.1, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-5)
gpr.fit(eval_grid, logpdf)
gpr.kernel_

# Now use GP-UKF to transform this into Gaussian on prob
mu_post, K_post = gp_ukf(gpr, xgrid, warp_func)
# Convert to scalars since in this case transform gives a scalar
mu_post = mu_post.item()
K_post = K_post.item()

# Give final estimation
CI = (mu_post - 1.96 * np.sqrt(K_post), mu_post + 1.96 * np.sqrt(K_post))
print(f"P ~ N({mu_post}, {np.sqrt(K_post)}) => CI: {CI}")

xs = np.random.multivariate_normal(mvn_mu, mvn_cov, size=10 ** 5)
prob_mc = np.mean(np.all((lb <= xs) & (xs <= ub), axis=1))

bmc_tail_prob = norm.cdf(prob_mc, loc=mu_post, scale=np.sqrt(K_post))
bmc_tail_prob = 2 * np.minimum(bmc_tail_prob, 1.0 - bmc_tail_prob)

print(f"MC estimate: {prob_mc}, BMC CI: {CI}")
print(f"tail prob: {bmc_tail_prob}")

prob_exact, _ = mvn.mvnun(lb + np.zeros(d), ub + np.zeros(d), mvn_mu, mvn_cov)

bmc_tail_prob = norm.cdf(prob_exact, loc=mu_post, scale=np.sqrt(K_post))
bmc_tail_prob = 2 * np.minimum(bmc_tail_prob, 1.0 - bmc_tail_prob)

print(f"Exact estimate mvnun: {prob_exact}, BMC CI: {CI}")
print(f"tail prob: {bmc_tail_prob}")

bmc_tail_prob = norm.cdf(prob_full, loc=mu_post, scale=np.sqrt(K_post))
bmc_tail_prob = 2 * np.minimum(bmc_tail_prob, 1.0 - bmc_tail_prob)

print(f"Exact estimate on same grid: {prob_full}, BMC CI: {CI}")
print(f"tail prob: {bmc_tail_prob}")

xsec_grid = xgrid[np.all(xgrid[:, 1:] == lb, axis=1), :]

# +
mu_prior, K_prior = gpr.predict(xsec_grid, return_std=False, return_cov=True)
LB = mu_prior - 1.96 * np.sqrt(np.diag(K_prior))
UB = mu_prior + 1.96 * np.sqrt(np.diag(K_prior))

logpdf_true = multivariate_normal.logpdf(xsec_grid, mvn_mu, mvn_cov)

# +
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

scalar_grid = xsec_grid[:, 0]

ax1.fill(
    np.concatenate([scalar_grid, scalar_grid[::-1]]),
    np.concatenate([LB, UB[::-1]]),
    alpha=0.25,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
ax1.plot(scalar_grid, logpdf_true, "r-")
ax1.plot(scalar_grid, mu_prior, "k")
ax1.set_xlim(scalar_grid[0], scalar_grid[-1])
ax1.grid("on")
ax1.set_ylabel("log lik")

ax2.fill(
    np.concatenate([scalar_grid, scalar_grid[::-1]]),
    np.concatenate([np.exp(LB), np.exp(UB[::-1])]),
    alpha=0.25,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
ax2.plot(scalar_grid, np.exp(logpdf_true), "r-")
ax2.plot(scalar_grid, np.exp(mu_prior), "k")
ax2.grid("on")
ax2.set_xlabel("$x_0$")
ax2.set_ylabel("lik")

plt.tight_layout()
# -
