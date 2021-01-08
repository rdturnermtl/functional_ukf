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

import numpy as np
from gp_ukf_core import big_ut
from sklearn.gaussian_process import kernels

np.random.seed(0)

n = 100
xform_n = 99

# +
# Define domain and build kernel
xgrid = np.linspace(0.0, 1.0, n)

# random mean func
mu = np.random.randn(n)

# kernel = kernels.Matern(length_scale=0.1, nu=3.5)
kernel = kernels.RBF(length_scale=0.1)
K = kernel(xgrid[:, None])
# -

# Parameter for random warp
W = np.random.randn(n, xform_n)
b = np.random.randn(xform_n)


def fx(x):
    return np.matmul(x, W) + b


# get exact answers since linear
mu_post_exact = np.matmul(W.T, mu) + b
K_post_exact = np.matmul(np.matmul(W.T, K), W)

# Get GP UKF version
mu_post, K_post = big_ut(mu, K, fx)

np.max(np.abs(mu_post - mu_post_exact))

np.max(np.abs(K_post - K_post_exact))
