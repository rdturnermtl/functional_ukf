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


# Setup problem
op = pde_operator()
xgrid = op.grid

# Setup GP
kernel = ConstantKernel(0.05, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False, alpha=1e-5)

init = only(gpr.sample_y(xgrid[:, None], 1, random_state=123).T)
ex_input = gpr.sample_y(xgrid[:, None], 5, random_state=456).T

plt.plot(xgrid, np.exp(ex_input.T), "--")
plt.plot(xgrid, np.exp(init), "k")

actual_out = only(op.forward(init[None, :]))
ex_out = op.forward(ex_input)

plt.plot(xgrid, np.exp(ex_out.T), "--")
plt.plot(xgrid, np.exp(actual_out), "k")

_, storage = op.pde_solve(init, T=100)

plot_kymograph(storage)  # visualize the result in a space-time plot

gpr.fit(xgrid[:, None][::10], init[::10])
gpr.kernel_

ex_input = gpr.sample_y(xgrid[:, None], 5, random_state=456).T

plt.plot(xgrid, np.exp(ex_input.T), "--")
plt.plot(xgrid, np.exp(init), "k")

# Now use GP-UKF to transform this into Gaussian on prob
mu_post, K_post = gp_ukf(gpr, xgrid[:, None], op.forward)

# +
# TODO cleanup plot
xgrid = op.grid

plt.plot(xgrid, np.exp(mu_post), "k")
plt.plot(xgrid, np.exp(mu_post - 1.96 * np.sqrt(np.diag(K_post))), "k--")
plt.plot(xgrid, np.exp(mu_post + 1.96 * np.sqrt(np.diag(K_post))), "k--")
plt.plot(xgrid, np.exp(actual_out), "r")
