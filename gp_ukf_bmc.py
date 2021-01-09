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
from scipy.stats import beta, binom
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

np.random.seed(0)

# +
n = 500

# model sampling
alpha0 = 1.0
beta0 = 1.0
n_data = 10
x_sum = 6

p_grid = np.linspace(0.0, 1.0, n)
p_eval = np.linspace(0.01, 0.99, 10)


# -


def log_integrand(p):
    logpdf = beta.logpdf(p, alpha0, beta0) + binom.logpmf(x_sum, n_data, p)
    return logpdf


def warp_func(loglik_surface):
    n_pts, n_grid = loglik_surface.shape
    assert p_grid.shape == (n_grid,)

    delta_p = np.diff(p_grid)
    assert delta_p.shape == (n_grid - 1,)

    lik_surface = np.exp(loglik_surface)
    int_val = np.sum(lik_surface[:, :-1] * delta_p[None, :], axis=1, keepdims=True)
    assert int_val.shape == (n_pts, 1)
    return int_val


# +
logpdf = log_integrand(p_eval)

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-5)  # TODO kernel etc
gpr.fit(p_eval[:, None], logpdf)
# -

mu_post, K_post = gp_ukf(gpr, p_grid[:, None], warp_func)

mu_post

K_post

(mu_post.item() - 1.96 * np.sqrt(K_post.item()), mu_post.item() + 1.96 * np.sqrt(K_post.item()))

mu_prior, K_prior = gpr.predict(p_grid[:, None], return_std=False, return_cov=True)


plt.plot(p_grid, mu_prior)
plt.plot(p_grid, mu_prior + 1.96 * np.sqrt(np.diag(K_prior)), "k--")
plt.plot(p_grid, mu_prior - 1.96 * np.sqrt(np.diag(K_prior)), "k--")
plt.plot(p_eval, logpdf, ".")

p_sample = np.random.rand(10000)

est = np.mean(np.exp(log_integrand(p_sample)))

est
