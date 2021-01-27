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
from joblib import Memory
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

np.random.seed(0)

memory = Memory(location=".")


def only(L):
    el, = L
    return el


init = np.log(np.ones(64) - np.sort(0.2 * np.random.randn(64)))


class pde_operator(object):
    dt = 1e-3
    tracker_dt = 1.0

    def __init__(self, n_grid=64, bounds=(-5.0, 5.0)):
        # Generate grid
        self._grid = CartesianGrid([list(bounds)], n_grid)
        self.grid = only(self._grid.axes_coords)

        # Define the PDE
        diffusivity = "1.01 + tanh(x)"
        term_1 = f"({diffusivity}) * laplace(c)"
        term_2 = f"dot(gradient({diffusivity}), gradient(c))"
        self._eq = PDE({"c": f"{term_1} + {term_2}"}, bc={"value": 0})

        # Setup cache so when we call center sigma point many times it is faster
        self.pde_solve = memory.cache(self.pde_solve, ignore=["self"])

    def pde_solve(self, input_state, T=1):
        # Store intermediate information of the simulation
        storage = MemoryStorage()

        # Input transform
        input_state = np.exp(input_state)

        # Setup, run solver, and extract result
        field = ScalarField(self._grid, input_state)
        res = self._eq.solve(field, T, dt=self.dt, tracker=storage.tracker(self.tracker_dt))
        assert isinstance(res, ScalarField)
        assert np.array_equal(only(res.grid.axes_coords), self.grid)
        output_state = res.data

        # Output transform
        output_state = np.log(output_state)
        return output_state, storage

    def forward(self, state):
        n_pts, n_grid = state.shape
        assert self.grid.shape == (n_grid,)

        all_res = np.zeros_like(state)
        for ii in range(n_pts):
            all_res[ii, :], _ = self.pde_solve(state[ii, :])
        return all_res


op = pde_operator()

_, storage = op.pde_solve(init, T=100)

plot_kymograph(storage)  # visualize the result in a space-time plot

# +
# Setup and train GP to the observations on the nrg
# TODO just use prior in filter, sample init from prior
# use GP on the log
grid_np = op.grid

# TODO use fixed hypers: 0.981**2 * RBF(length_scale=0.691)
#   sample init from these hypers instead
#   what to use as obs of init??
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(0.1, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-5)
gpr.fit(grid_np[:, None][::3], init[::3])
gpr.kernel_
# -

# Now use GP-UKF to transform this into Gaussian on prob
mu_post, K_post = gp_ukf(gpr, grid_np[:, None], op.forward)

out1, = op.forward(init[None, :])

# +
# TODO cleanup plot
xgrid = op.grid

plt.plot(xgrid, np.exp(init), "g")
plt.plot(xgrid, np.exp(mu_post), "k")
plt.plot(xgrid, np.exp(mu_post - 1.96 * np.sqrt(np.diag(K_post))), "k--")
plt.plot(xgrid, np.exp(mu_post + 1.96 * np.sqrt(np.diag(K_post))), "k--")
plt.plot(xgrid, np.exp(out1), "r")
# -

# TODO plot init with sigma in, add sigma out to above plot
np.var(init)

kernel = ConstantKernel(0.05, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-5)
yy = gpr.sample_y(grid_np[:, None], 3)

plt.plot(grid_np, np.exp(yy), "--")
plt.plot(grid_np, np.exp(init), "k")

yy2 = op.forward(yy.T)

plt.plot(grid_np, np.exp(yy2).T, "--")
plt.plot(grid_np, np.exp(out1), "k")
