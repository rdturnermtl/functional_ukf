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

# +
import numpy as np
from gp_ukf_core import gp_ukf
from scipy.stats import multivariate_normal, mvn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# -

np.random.seed(0)

# +
d = 2
n_grid = 50

xgrid = np.linspace(0.0, 1.0, n_grid + 1)
dx = np.median(np.diff(xgrid))
xgrid = np.array(list(product(xgrid[:-1], repeat=d)))
# -

xgrid.shape

dx

mvn.mvnun(np.zeros(d), np.ones(d), np.zeros(d), np.eye(d))

xs = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=100000)
np.mean(np.all((0.0 <= xs) & (xs <= 1.0), axis=1))

np.sum(np.exp(multivariate_normal.logpdf(xgrid, np.zeros(d), np.eye(d)))) * (dx ** d)


def warp_func(loglik_surface):
    n_pts, n_grid = loglik_surface.shape
    assert xgrid.shape == (n_grid, d)

    lik_surface = np.exp(loglik_surface)
    int_val = np.sum(lik_surface, axis=1, keepdims=True) * (dx ** d)
    assert int_val.shape == (n_pts, 1)
    return int_val


warp_func(multivariate_normal.logpdf(xgrid, np.zeros(d), np.eye(d))[None, :])

# +
n_eval_grid = 8

eval_grid = np.linspace(0.0, 1.0, n_eval_grid + 1)
eval_grid = np.array(list(product(eval_grid[:-1], repeat=d)))

logpdf = multivariate_normal.logpdf(eval_grid, np.zeros(d), np.eye(d))
# -

# Setup and train GP to the observations on the nrg
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(0.1, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-5)
gpr.fit(eval_grid, logpdf)
gpr.kernel_

# Now use GP-UKF to transform this into Gaussian on marginal lik
mu_post, K_post = gp_ukf(gpr, xgrid, warp_func)
# Convert to scalars since in this case transform gives a scalar
mu_post = mu_post.item()
K_post = K_post.item()

# Give final estimation
CI = (mu_post - 1.96 * np.sqrt(K_post), mu_post + 1.96 * np.sqrt(K_post))
print(f"marglik ~ N({mu_post}, {np.sqrt(K_post)}) => CI: {CI}")
