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


# +
import numpy as np
from gp_ukf_core import gp_ukf
from joblib import Memory
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# -

memory = Memory(location=".")

np.random.seed(0)

init = np.ones(64) - np.sort(0.2 * np.random.randn(64))

# Expanded definition of the PDE
diffusivity = "1.01 + tanh(x)"
term_1 = f"({diffusivity}) * laplace(c)"
term_2 = f"dot(gradient({diffusivity}), gradient(c))"
eq = PDE({"c": f"{term_1} + {term_2}"}, bc={"value": 0})

grid = CartesianGrid([[-5, 5]], 64)  # generate grid
field = ScalarField(grid, init)  # generate initial condition
grip_np, = grid.axes_coords

storage = MemoryStorage()  # store intermediate information of the simulation
res = eq.solve(field, 100, dt=1e-3, tracker=storage.tracker(1))  # solve the PDE

plot_kymograph(storage)  # visualize the result in a space-time plot


def state_to_vec(state):
    assert isinstance(state, ScalarField)

    data = state.data
    grid, = res.grid.axes_coords

    n_grid, = data.shape
    assert grid.shape == (n_grid,)

    return data, grid


@memory.cache
def warp_func(state):
    n_pts, n_grid = state.shape
    # TODO assert compatible with grid

    all_res = np.zeros_like(state)
    for ii in range(n_pts):
        # TODO need to put memory around inside of the loop
        field = ScalarField(grid, state[ii, :])  # generate initial condition
        res = eq.solve(field, 1.0, dt=1e-3)
        data, _ = state_to_vec(res)
        # TODO assert some grid stuff
        all_res[ii, :] = data
    return all_res


# Setup and train GP to the observations on the nrg
# TODO just use prior in filter, sample init from prior
# use GP on the log
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(0.1, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-5)
gpr.fit(grip_np[:, None][::3], init[::3])
gpr.kernel_

# Now use GP-UKF to transform this into Gaussian on prob
mu_post, K_post = gp_ukf(gpr, grip_np[:, None], warp_func)

mu_post

np.diag(K_post)

storage.data

init

out1 = warp_func(init[None, :])

storage.data[1] - out1
