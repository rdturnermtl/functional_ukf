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

from sklearn.gaussian_process import kernels
import numpy as np

# kernel = kernels.Matern(length_scale=0.1, nu=3.5)
kernel = kernels.RBF(length_scale=0.1)

xgrid = np.linspace(0,1,100)

K = kernel(xgrid[:, None])

K.shape

K

u, s, vh = np.linalg.svd(K)

np.max(np.abs(np.matmul(np.matmul(u, np.diag(s)), vh) - K))

np.max(np.abs(np.matmul(np.matmul(u, np.diag(np.sqrt(s))), np.matmul(np.diag(np.sqrt(s)), vh)) - K))

np.max(np.abs(np.matmul(u, np.diag(np.sqrt(s))) - np.matmul(np.diag(np.sqrt(s)), vh).T))

sqrt_K = np.matmul(u, np.diag(np.sqrt(s)))

np.max(np.abs(np.matmul(sqrt_K, sqrt_K.T) - K))


def sqrt_U(K):
    u, s, vh = np.linalg.svd(K)
    sqrt_K = np.matmul(u, np.diag(np.sqrt(s))).T
    return sqrt_K


sqrt_K = sqrt_U(K)
np.max(np.abs(np.matmul(sqrt_K.T, sqrt_K) - K))

import matplotlib.pyplot as plt

plt.plot(sqrt_K[-3, :], '.-')

sqrt_K_crop = sqrt_K + 0.0
sqrt_K_crop[20:, :] = 0.0

np.max(np.abs(np.matmul(sqrt_K_crop.T, sqrt_K_crop) - K))

plt.plot(sqrt_K_crop.T, '.-')

from filterpy.kalman.unscented_transform import unscented_transform
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints

points = MerweScaledSigmaPoints(n=len(xgrid), alpha=1e-3, beta=2.0, kappa=0.0, sqrt_method=sqrt_U)
Wm = points.Wm
Wc = points.Wc

sigma_points = points.sigma_points(np.zeros_like(xgrid), K)

sigma_points.shape

W = np.random.randn(100, 99)
b = np.random.randn(99)
def fx(x):
    return np.matmul(x, W) + b


transformed_sigma_points = fx(sigma_points)

transformed_sigma_points.shape

mu_post, K_post = unscented_transform(transformed_sigma_points, Wm, Wc)

np.max(np.abs(mu_post - b))

np.max(np.abs(K_post - np.matmul(np.matmul(W.T, K), W)))


def sqrt_U_approx(K):
    u, s, vh = np.linalg.svd(K)
    sqrt_K = np.matmul(u, np.diag(np.sqrt(s))).T
    sqrt_K[20:, :] = 0.0
    return sqrt_K


points = MerweScaledSigmaPoints(n=len(xgrid), alpha=1e-3, beta=2.0, kappa=0.0, sqrt_method=sqrt_U_approx)
Wm = points.Wm
Wc = points.Wc

sigma_points = points.sigma_points(np.zeros_like(xgrid), K)

transformed_sigma_points = fx(sigma_points)

mu_post, K_post = unscented_transform(transformed_sigma_points, Wm, Wc)

np.max(np.abs(mu_post - b))

np.max(np.abs(K_post - np.matmul(np.matmul(W.T, K), W)))


