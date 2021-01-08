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

# +
import matplotlib.pyplot as plt

import numpy as np
from gp_ukf_core import big_ut
from sklearn.gaussian_process import kernels
from scipy.stats import norm
# -

np.random.seed(0)

# +
n = 100

# Try warping thru exp
warp_func = np.exp

# +
# Define domain and build kernel
xgrid = np.linspace(0.0, 1.0, n)

# random mean func
mu = 1 + np.sort(np.random.randn(n) / 10)

# kernel = kernels.Matern(length_scale=0.1, nu=3.5)
kernel = kernels.RBF(length_scale=0.1)
K = kernel(xgrid[:, None])

# +
std = np.sqrt(np.diag(K))
LB, UB = norm.interval(0.95, loc=mu, scale=std)

y = np.random.multivariate_normal(mu, K, size=5)
# -

plt.fill(np.concatenate([xgrid, xgrid[::-1]]),
         np.concatenate([np.exp(LB), np.exp(UB[::-1])]),
         alpha=.25, fc='b', ec='None', label='95% confidence interval')
plt.plot(xgrid, warp_func(y.T))

# Get GP UKF version
mu_post, K_post = big_ut(mu, K, fx=warp_func)

# +
std_post = np.sqrt(np.diag(K_post))
LB, UB = norm.interval(0.95, loc=mu_post, scale=std_post)

y = np.random.multivariate_normal(mu_post, K_post, size=5)
# -

plt.fill(np.concatenate([xgrid, xgrid[::-1]]),
         np.concatenate([LB, UB[::-1]]),
         alpha=.25, fc='b', ec='None', label='95% confidence interval')
plt.plot(xgrid, y.T)

# +
# TODO MC version

# +
# TODO plot all side-by-side shared-y

# then do sum-exp, or log-sum-exp, and look at distn

# re-interp sigma with gp again and then get exact integral of that
#    make that sigma func
# test on case of actual marg lik test case: norm-norm model or beta-bern
