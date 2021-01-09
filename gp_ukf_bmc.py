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

import matplotlib.pyplot as plt
import numpy as np
from gp_ukf_core import gp_ukf
from scipy.special import betaln
from scipy.stats import beta, binom, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

np.random.seed(0)

# +
# Parameters for the model we want marginal likelihood of
alpha0 = 1.0
beta0 = 1.0
n_data = 10

# Sufficient statistics of the data
x_sum = 6

# How many points can we evaluate the nrg
n_nrg = 10

# How many points will we interp with GP in functional-UKF
n_interp = 500


# -


# Setup grids
p_grid = np.linspace(0.0, 1.0, n_interp)
p_eval = np.linspace(0.01, 0.99, n_nrg)


def log_integrand(p):
    logpdf = beta.logpdf(p, alpha0, beta0) + binom.logpmf(x_sum, n_data, p)
    return logpdf


# +
# Also get get exact loglik for ground truth
def binomln(n, k):
    # Assumes binom(n, k) >= 0
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def marg_loglik(k, n, alpha0, beta0):
    loglik = binomln(n, k) + (betaln(k + alpha0, (n - k) + beta0) - betaln(alpha0, beta0))
    return loglik


# -

# Get the "data" about the nrg for BMC
logpdf = log_integrand(p_eval)


def warp_func(loglik_surface):
    n_pts, n_grid = loglik_surface.shape
    assert p_grid.shape == (n_grid,)

    delta_p = np.diff(p_grid)
    assert delta_p.shape == (n_grid - 1,)

    lik_surface = np.exp(loglik_surface)
    int_val = np.sum(lik_surface[:, :-1] * delta_p[None, :], axis=1, keepdims=True)
    assert int_val.shape == (n_pts, 1)
    return int_val


# Setup and train GP to the observations on the nrg
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-5)
gpr.fit(p_eval[:, None], logpdf)
gpr.kernel_

# Now use GP-UKF to transform this into Gaussian on marginal lik
mu_post, K_post = gp_ukf(gpr, p_grid[:, None], warp_func)
# Convert to scalars since in this case transform gives a scalar
mu_post = mu_post.item()
K_post = K_post.item()

# Give final estimation
CI = (mu_post - 1.96 * np.sqrt(K_post), mu_post + 1.96 * np.sqrt(K_post))
print(f"marglik ~ N({mu_post}, {np.sqrt(K_post)}) => CI: {CI}")

# +
# Now let's gets some more variables for visualizations
mu_prior, K_prior = gpr.predict(p_grid[:, None], return_std=False, return_cov=True)
LB = mu_prior - 1.96 * np.sqrt(np.diag(K_prior))
UB = mu_prior + 1.96 * np.sqrt(np.diag(K_prior))

logpdf_true = log_integrand(p_grid)


# +
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

ax1.fill(
    np.concatenate([p_grid, p_grid[::-1]]),
    np.concatenate([LB, UB[::-1]]),
    alpha=0.25,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
ax1.plot(p_grid, logpdf_true, "r-")
ax1.plot(p_grid, mu_prior, "k")
ax1.plot(p_eval, logpdf, ".")
ax1.set_xlim(p_grid[0], p_grid[-1])
ax1.grid("on")
ax1.set_ylabel("log lik")

ax2.fill(
    np.concatenate([p_grid, p_grid[::-1]]),
    np.concatenate([np.exp(LB), np.exp(UB[::-1])]),
    alpha=0.25,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
ax2.plot(p_grid, np.exp(logpdf_true), "r-")
ax2.plot(p_grid, np.exp(mu_prior), "k")
ax2.plot(p_eval, np.exp(logpdf), ".")
ax2.grid("on")
ax2.set_xlabel("p")
ax2.set_ylabel("lik")

plt.tight_layout()

# +
# Now let's do MC estimate of integral with big N
p_sample = np.random.rand(10 ** 5)
mc_estimate = np.mean(np.exp(log_integrand(p_sample)))

bmc_tail_prob = norm.cdf(mc_estimate, loc=mu_post, scale=np.sqrt(K_post))
bmc_tail_prob = 2 * np.minimum(bmc_tail_prob, 1.0 - bmc_tail_prob)
# -

print(f"MC estimate: {mc_estimate}, BMC CI: {CI}")
print(f"tail prob: {bmc_tail_prob}")

# Now we can test againt exact
exact_lik = np.exp(marg_loglik(x_sum, n_data, alpha0, beta0))
bmc_tail_prob = norm.cdf(exact_lik, loc=mu_post, scale=np.sqrt(K_post))
bmc_tail_prob = 2 * np.minimum(bmc_tail_prob, 1.0 - bmc_tail_prob)

print(f"exact: {exact_lik}, BMC CI: {CI}")
print(f"tail prob: {bmc_tail_prob}")
