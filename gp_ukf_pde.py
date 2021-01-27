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

memory = Memory(location=".")

np.random.seed(0)

init = np.ones(64) - np.sort(0.2 * np.random.randn(64))


# +
def only(L):
    el, = L
    return el


class pde_operator(object):
    def __init__(self, n_grid=64):
        # TODO store the np grid instead
        self.n_grid = n_grid
        self.grid = CartesianGrid([[-5, 5]], n_grid)  # generate grid

        # Expanded definition of the PDE
        diffusivity = "1.01 + tanh(x)"
        term_1 = f"({diffusivity}) * laplace(c)"
        term_2 = f"dot(gradient({diffusivity}), gradient(c))"
        self.eq = PDE({"c": f"{term_1} + {term_2}"}, bc={"value": 0})

        self.pde_solve = memory.cache(self.pde_solve, ignore=["self"])

        x_grid, = self.grid.axes_coords
        assert x_grid.shape == (n_grid,)

    def pde_solve(self, state, T=1):
        storage = MemoryStorage()  # store intermediate information of the simulation

        # TODO experiment with exp - f - log xform on state in this func

        field = ScalarField(self.grid, state)  # generate initial condition

        # TODO make class level consts
        res = self.eq.solve(field, T, dt=1e-3, tracker=storage.tracker(1))
        assert isinstance(res, ScalarField)
        assert np.array_equal(only(res.grid.axes_coords), only(self.grid.axes_coords))
        data = res.data
        return data, storage

    def forward(self, state):
        n_pts, n_grid = state.shape
        assert n_grid == self.n_grid

        all_res = np.zeros_like(state)
        for ii in range(n_pts):
            all_res[ii, :], _ = self.pde_solve(state[ii, :])
        return all_res


# -

op = pde_operator()

_, storage = op.pde_solve(init, T=100)

plot_kymograph(storage)  # visualize the result in a space-time plot

# +
# Setup and train GP to the observations on the nrg
# TODO just use prior in filter, sample init from prior
# use GP on the log
grid_np, = op.grid.axes_coords

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

xgrid, = op.grid.axes_coords

plt.plot(xgrid, init, "g")
plt.plot(xgrid, mu_post, "k")
plt.plot(xgrid, mu_post - 1.96 * np.sqrt(np.diag(K_post)), "k--")
plt.plot(xgrid, mu_post + 1.96 * np.sqrt(np.diag(K_post)), "k--")
plt.plot(xgrid, out1, "r")

# +
# TODO plot init with sigma in, add sigma out to above plot
