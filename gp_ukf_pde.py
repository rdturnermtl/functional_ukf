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
from gp_ukf_core import gp_sigma_points, gp_ukf
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
    tracker_dt = 1e-2

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

# +
# Sample an initial condition for PDE
actual_input = only(gpr.sample_y(xgrid[:, None], 1, random_state=123).T)

# Also get the output state for the initial condition
actual_output = only(op.forward(actual_input[None, :]))
# -

# Setup GP prior with some limit observations of initial PDE state
input_obs_points, input_obs = xgrid[::10], actual_input[::10]
gpr.fit(input_obs_points[:, None], input_obs)

# # Diffusion of actual input

_, storage = op.pde_solve(actual_input)

plot_kymograph(storage)  # visualize the result in a space-time plot

# # Uncertainty on input state

example_input = gpr.sample_y(xgrid[:, None], 5, random_state=456).T

plt.plot(xgrid, np.exp(example_input.T), "--")
plt.plot(xgrid, np.exp(actual_input), "k")
plt.plot(input_obs_points, np.exp(input_obs), "ro")

# # Uncertainty on output state

example_output = op.forward(example_input)

plt.plot(xgrid, np.exp(example_output.T), "--")
plt.plot(xgrid, np.exp(actual_output), "k")

# # UT on input

sigma_points = gp_sigma_points(gpr, xgrid[:, None], alpha=0.1)

plt.plot(xgrid, np.exp(sigma_points.T), "--")
plt.plot(xgrid, np.exp(actual_input), "k")
plt.plot(xgrid, np.exp(sigma_points[0]), "k--")
plt.plot(input_obs_points, np.exp(input_obs), "ro")

# # UT for output

sigma_points_out = op.forward(sigma_points)

plt.plot(xgrid, np.exp(sigma_points_out.T), "--")
plt.plot(xgrid, np.exp(sigma_points_out[0]), "k--")
plt.plot(xgrid, np.exp(actual_output), "k")

# # Putting it all together

# Now use GP-UKF to transform this into Gaussian on prob
mu_post, K_post = gp_ukf(gpr, xgrid[:, None], op.forward, alpha=0.1)

# +
# TODO cleanup plot
xgrid = op.grid

plt.plot(xgrid, np.exp(mu_post), "k")
plt.plot(xgrid, np.exp(mu_post - 1.96 * np.sqrt(np.diag(K_post))), "k--")
plt.plot(xgrid, np.exp(mu_post + 1.96 * np.sqrt(np.diag(K_post))), "k--")
plt.plot(xgrid, np.exp(actual_output), "r")
