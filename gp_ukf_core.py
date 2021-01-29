import numpy as np
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from filterpy.kalman.unscented_transform import unscented_transform
from sklearn.gaussian_process import GaussianProcessRegressor


def sqrt_U_approx(K, eig_thold=1e-6):
    u, s, vh = np.linalg.svd(K)
    sqrt_K = np.matmul(u, np.diag(np.sqrt(s))).T

    assert np.isclose(s[0], np.max(s))
    crop = (s / s[0]) < eig_thold
    # TODO switch to logging package
    print(f"Zero out {np.sum(crop)} / {crop.size} rows.")

    sqrt_K[crop, :] = 0.0
    return sqrt_K


def big_ut(mu_prior, K_prior, fx, *, alpha=1e-3, beta=2.0, kappa=0.0):
    n, = mu_prior.shape
    assert K_prior.shape == (n, n)

    points = MerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa, sqrt_method=sqrt_U_approx)
    Wm = points.Wm
    Wc = points.Wc

    sigma_points = points.sigma_points(mu_prior, K_prior)

    # TODO consider memoizer wrapper than remembers mu_prior
    transformed_sigma_points = fx(sigma_points)

    mu_post, K_post = unscented_transform(transformed_sigma_points, Wm, Wc)
    return mu_post, K_post


def gp_ukf(gpr, xgrid, fx, *, alpha=1e-3, beta=2.0, kappa=0.0):
    assert isinstance(gpr, GaussianProcessRegressor)
    n, _ = xgrid.shape

    mu_prior, K_prior = gpr.predict(xgrid, return_std=False, return_cov=True)
    assert mu_prior.shape == (n,)
    assert K_prior.shape == (n, n)

    mu_post, K_post = big_ut(mu_prior, K_prior, fx, alpha=alpha, beta=beta, kappa=kappa)
    return mu_post, K_post


def gp_sigma_points(gpr, xgrid, *, alpha=1e-3, beta=2.0, kappa=0.0):
    mu_prior, K_prior = gpr.predict(xgrid, return_std=False, return_cov=True)
    n, = mu_prior.shape
    points = MerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa, sqrt_method=sqrt_U_approx)
    sigma_points = points.sigma_points(mu_prior, K_prior)
    return sigma_points


def gauss_update(x, idx, noise_var, mu_prior, K_prior):
    mu_1 = mu_prior
    mu_2 = mu_prior[idx]

    S11 = K_prior
    S12 = K_prior[:, idx]
    S22 = K_prior[idx, idx] + noise_var

    fac = (x - mu_2) / S22
    assert fac.shape == ()
    mu_post = mu_1 + S12 * fac
    K_diff = np.outer(S12, S12) / S22
    assert K_diff.shape == S11.shape
    K_post = S11 - K_diff

    return mu_post, K_post


def func_filter(x, idx, noise_var, mu_0, K_0, fx, *, alpha=1e-3, beta=2.0, kappa=0.0):
    n_grid, = mu_0.shape
    n_step, = x.shape
    assert idx.shape == (n_step,)
    assert np.all(n_step >= 0)
    assert np.all(n_step < n_grid)
    assert noise_var > 0
    assert K_0.shape == (n_grid, n_grid)

    mu_pre_obs = np.zeros((n_step, n_grid))
    K_pre_obs = np.zeros((n_step, n_grid, n_grid))
    mu_post_obs = np.zeros((n_step, n_grid))
    K_post_obs = np.zeros((n_step, n_grid, n_grid))

    mu_state = mu_0
    K_state = K_0
    for tt, (xx, ii) in enumerate(zip(x, idx)):
        mu_pre_obs[tt], K_pre_obs[tt] = big_ut(mu_state, K_state, fx, alpha=alpha, beta=beta, kappa=kappa)
        mu_post_obs[tt], K_post_obs[tt] = gauss_update(xx, ii, noise_var, mu_pre_obs[tt], K_pre_obs[tt])
        mu_state, K_state = mu_post_obs[tt], K_post_obs[tt]
    return mu_pre_obs, K_pre_obs, mu_post_obs, K_post_obs
