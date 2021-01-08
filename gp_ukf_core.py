from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from filterpy.kalman.unscented_transform import unscented_transform
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints


def sqrt_U_approx(K, eig_thold=1e-10):
    u, s, vh = np.linalg.svd(K)
    sqrt_K = np.matmul(u, np.diag(np.sqrt(s))).T

    assert np.isclose(s[0], np.max(s))
    crop = (s / s[0]) < eig_thold
    sqrt_K[crop, :] = 0.0
    return sqrt_K


def big_ut(mu_prior, K_prior, fx, *, alpha=1e-3, beta=2.0, kappa=0.0):
    n, = mu_prior.shape
    assert K_prior.shape (n, n)

    points = MerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa, sqrt_method=sqrt_U_approx)
    Wm = points.Wm
    Wc = points.Wc

    sigma_points = points.sigma_points(mu_prior, K_prior)

    # TODO consider memoizer wrapper than remembers mu_prior
    transformed_sigma_points = fx(sigma_points)

    mu_post, K_post = unscented_transform(transformed_sigma_points, Wm, Wc)
    return mu_post, K_post


def gp_ukf(gpr, xgrid, fx):
    assert isinstance(gpr, sklearn.gaussian_process.GaussianProcessRegressor)
    n, _ = xgrid.shape

    mu_prior, K_prior = gpr.predict(xgrid, return_std=False, return_cov=True)
    assert mu_prior.shape == (n,)
    assert K_prior.shape == (n, n)

    mu_post, K_post = big_ut(mu_prior, K_prior, fx)
    return mu_post, K_post
