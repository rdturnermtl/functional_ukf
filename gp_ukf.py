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
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from filterpy.kalman.unscented_transform import unscented_transform
from scipy.linalg import sqrtm
from sklearn.gaussian_process import kernels

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8

# kernel = kernels.Matern(length_scale=0.1, nu=3.5)
kernel = kernels.RBF(length_scale=0.1)

xgrid = np.linspace(0, 1, 100)

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


plt.plot(sqrt_K[-3, :], ".-")

sqrt_K_crop = sqrt_K + 0.0
sqrt_K_crop[20:, :] = 0.0

np.max(np.abs(np.matmul(sqrt_K_crop.T, sqrt_K_crop) - K))

plt.plot(sqrt_K_crop.T, ".-")


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

# +
_, s, _ = np.linalg.svd(K)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), dpi=150)

ax1.semilogy(1 + np.arange(len(s)), s)
ax1.set_xlim(0, len(s))
ax1.set_xlabel("row $i$")
ax1.set_ylabel("singular value $s$")
ax1.grid("on")

cax = ax2.matshow(K, vmin=-1, vmax=1, cmap=plt.cm.coolwarm)
ax2.set_xticks([])
ax2.set_yticks([])
fig.colorbar(cax)

plt.tight_layout(w_pad=0)

print(np.min(K), np.max(K))

# +
U = sqrt_U(K)
err = np.max(np.abs(np.matmul(U.T, U) - K))
print(np.min(U), np.max(U))
print(f"error: {err}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), dpi=150)

ax1.plot(xgrid, U[::9, :].T)
ax1.set_xlim(xgrid[0], xgrid[-1])
ax1.set_xlabel("$x$")
ax1.set_ylabel("sigma function")
ax1.grid("on")

cax = ax2.matshow(U, vmin=-1, vmax=1, cmap=plt.cm.coolwarm)
ax2.set_xticks([])
ax2.set_yticks([])
fig.colorbar(cax)

plt.tight_layout(w_pad=0)

# +
epsilon = 1e-10

U = np.linalg.cholesky(K + epsilon * np.eye(len(K))).T
err = np.max(np.abs(np.matmul(U.T, U) - K))
print(np.min(U), np.max(U))
print(f"error: {err}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), dpi=150)

ax1.plot(xgrid, U[::9, :].T)
ax1.set_xlim(xgrid[0], xgrid[-1])
ax1.set_xlabel("$x$")
ax1.set_ylabel("sigma function")
ax1.grid("on")

cax = ax2.matshow(U, vmin=-1, vmax=1, cmap=plt.cm.coolwarm)
ax2.set_xticks([])
ax2.set_yticks([])
fig.colorbar(cax)

plt.tight_layout(w_pad=0)

# +
U = sqrtm(K).real

err = np.max(np.abs(np.matmul(U.T, U) - K))
print(np.min(U), np.max(U))
print(f"error: {err}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), dpi=150)

ax1.plot(xgrid, U[::9, :].T)
ax1.set_xlim(xgrid[0], xgrid[-1])
ax1.set_xlabel("$x$")
ax1.set_ylabel("sigma function")
ax1.grid("on")

cax = ax2.matshow(U, vmin=-1, vmax=1, cmap=plt.cm.coolwarm)
ax2.set_xticks([])
ax2.set_yticks([])
fig.colorbar(cax)

plt.tight_layout(w_pad=0)
# -

u, s, vh = np.linalg.svd(K)
idx = np.argmax(s <= 1e-10)
u = u[:, :idx]
vh = u.T
s = s[:idx]

# +
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4), dpi=150)

ax1.matshow(u, vmin=-np.max(np.abs(u)), vmax=np.max(np.abs(u)), cmap=plt.cm.coolwarm)
ax1.set_xticks([])
ax1.set_yticks([])

ax2.matshow(np.diag(s), vmin=-np.max(np.abs(s)), vmax=np.max(np.abs(s)), cmap=plt.cm.coolwarm)
ax2.set_xticks([])
ax2.set_yticks([0.5 * len(s)])
ax2.set_yticklabels([r"$\times$"], fontsize=30)

ax3.matshow(vh, vmin=-np.max(np.abs(vh)), vmax=np.max(np.abs(vh)), cmap=plt.cm.coolwarm)
ax3.set_xticks([])
ax3.set_yticks([0.5 * len(vh)])
ax3.set_yticklabels([r"$\times$"], fontsize=30)

ax4.matshow(K, vmin=-1, vmax=1, cmap=plt.cm.coolwarm)
ax4.set_xticks([])
ax4.set_yticks([0.5 * len(K)])
ax4.set_yticklabels(["$=$"], fontsize=30)

plt.tight_layout(w_pad=0)
# -
