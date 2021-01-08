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
from gp_ukf_core import big_ut
from scipy.stats import norm
from sklearn.gaussian_process import kernels

np.random.seed(0)

# +
n = 500

# Try warping thru exp
warp_func = np.exp

# +
# Define domain and build kernel
xgrid = np.linspace(0.0, 1.0, n)

# random mean func
mu = np.sort(np.random.randn(n) / 10)

# kernel = kernels.Matern(length_scale=0.1, nu=3.5)
kernel = 0.05 * kernels.RBF(length_scale=0.1)
K = kernel(xgrid[:, None])

# +
# Get the exact/MC version
std = np.sqrt(np.diag(K))
LB, UB = norm.interval(0.95, loc=mu, scale=std)
LB = np.exp(LB)
UB = np.exp(UB)

rnd = np.random.RandomState(123)
y_exact = warp_func(rnd.multivariate_normal(mu, K, size=5))

# For exp this could be done analytically, but doing MC for generality
mu_mc = np.mean(warp_func(np.random.multivariate_normal(mu, K, size=1000)), axis=0)
# -

# Get GP UKF version
mu_post, K_post = big_ut(mu, K, fx=warp_func)

# +
# For plotting from UKF version
std_post = np.sqrt(np.diag(K_post))
LB_post, UB_post = norm.interval(0.95, loc=mu_post, scale=std_post)

rnd = np.random.RandomState(123)
y_post = rnd.multivariate_normal(mu_post, K_post, size=5)

# +
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))

ax1.fill(
    np.concatenate([xgrid, xgrid[::-1]]),
    np.concatenate([LB, UB[::-1]]),
    alpha=0.25,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
ax1.plot(xgrid, mu_mc, 'k')
ax1.plot(xgrid, y_exact.T)
ax1.set_xlim(xgrid[0], xgrid[-1])
ax1.grid("on")

ax2.fill(
    np.concatenate([xgrid, xgrid[::-1]]),
    np.concatenate([LB_post, UB_post[::-1]]),
    alpha=0.25,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
ax2.plot(xgrid, mu_post, "k")
ax2.plot(xgrid, y_post.T)
ax2.set_xlim(xgrid[0], xgrid[-1])
ax2.grid("on")

plt.tight_layout()
# -


