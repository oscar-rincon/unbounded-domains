
import os
import numpy as np
import torch
import sympy as sp
import matplotlib.pyplot as plt
from datetime import datetime

def source_term_inf(xs_inf, ys_inf, alpha_s_inf, beta_s_inf, eps_s_inf):


    u = sp.exp(
        -alpha_s_inf*(xs_inf**2 + ys_inf**2)
    ) * sp.cos(beta_s_inf*ys_inf)

    k = 1 + 2/(1 + sp.exp(-ys_inf/eps_s_inf))

    # -------------------------
    # Poisson source term
    # -------------------------
    ux = sp.diff(u, xs_inf)
    uy = sp.diff(u, ys_inf)

    f = -(
        sp.diff(k*ux, xs_inf)
        +
        sp.diff(k*uy, ys_inf)
    )

    f = sp.simplify(f)

    f_numpy = sp.lambdify(
        (xs_inf, ys_inf, alpha_s_inf, beta_s_inf, eps_s_inf),
        f,
        "numpy",
    )

    return f_numpy
 
def analytical_solution_inf(xs_inf, ys_inf, alpha_s_inf, beta_s_inf):
    """
    Analytical solution u(x,y).
    """
    return np.exp(-alpha_s_inf * (xs_inf**2 + ys_inf**2)) * np.cos(beta_s_inf * ys_inf)

def coefficient_inf(xs_inf, ys_inf, epsilon):
    """
    Variable coefficient k(y).
    """
    return 1 + 2 / (1 + np.exp(-ys_inf / epsilon))


def generate_dataset_inf(
    alpha=0.5,
    beta=5.0,
    epsilon=1.0,
    domain=(-5.0, 5.0),
    n_obs_u=100,
    n_obs_k=100,
    n_pde=10_000,
    n_grid=300,
    sampling="uniform",   # "uniform", "gaussian", "gaussian_exponential"
    sigma=2.0,
    exp_scale=1.0,
    device="cpu",
    dtype=torch.float32,
    plot=False,
):
    xmin, xmax = domain

    # --------------------------------------------------
    # Sampling functions
    # --------------------------------------------------
    def truncated_gaussian(n):
        x = np.random.normal(0.0, sigma, 5 * n)
        x = x[(x >= xmin) & (x <= xmax)]
        while len(x) < n:
            extra = np.random.normal(0.0, sigma, 2 * n)
            extra = extra[(extra >= xmin) & (extra <= xmax)]
            x = np.concatenate((x, extra))
        return x[:n]

    def truncated_exponential(n):
        # Symmetric exponential (Laplace)
        y = np.random.laplace(0.0, exp_scale, 3 * n)
        y = y[(y >= xmin) & (y <= xmax)]
        while len(y) < n:
            extra = np.random.laplace(0.0, exp_scale, 2 * n)
            extra = extra[(extra >= xmin) & (extra <=xmax)]
            y = np.concatenate((y, extra))
        return y[:n]

    def sample_points(n):
        if sampling == "uniform":
            x = np.random.uniform(xmin, xmax, n)
            y = np.random.uniform(xmin, xmax, n)

        elif sampling == "gaussian":
            x = truncated_gaussian(n)
            y = truncated_gaussian(n)

        elif sampling == "gaussian_exponential":
            x = truncated_gaussian(n)
            y = truncated_exponential(n)

        else:
            raise ValueError(f"Unknown sampling '{sampling}'")

        return x, y

    # --------------------------------------------------
    # Symbolic variables
    # --------------------------------------------------
    xs, ys = sp.symbols("x y")
    alpha_s, beta_s = sp.symbols("alpha beta", positive=True)
    eps_s = sp.symbols("epsilon", positive=True)

    # --------------------------------------------------
    # Observation points
    # --------------------------------------------------
    x_obs, y_obs = sample_points(n_obs_u)

    # Keep k observations uniform
    #x_obs_k = np.random.uniform(xmin, xmax, n_obs_k)
    #y_obs_k = np.random.uniform(xmin, xmax, n_obs_k)
    x_obs_k, y_obs_k = sample_points(n_obs_k)

    u_obs = analytical_solution_inf(x_obs, y_obs, alpha, beta)
    k_obs = coefficient_inf(x_obs_k, y_obs_k, epsilon)

    # --------------------------------------------------
    # PDE collocation points
    # --------------------------------------------------
    x_pde, y_pde = sample_points(n_pde)

    f_pde = source_term_inf(xs, ys, alpha_s, beta_s, eps_s)
    f_values = f_pde(x_pde, y_pde, alpha, beta, epsilon)

    # --------------------------------------------------
    # Visualization grid
    # --------------------------------------------------
    x = np.linspace(xmin, xmax, n_grid)
    y = np.linspace(xmin, xmax, n_grid)
    X, Y = np.meshgrid(x, y)
    U = analytical_solution_inf(X, Y, alpha, beta)

    # --------------------------------------------------
    # Torch tensors
    # --------------------------------------------------
    X_obs = torch.tensor(
        np.column_stack((x_obs, y_obs)),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    X_obs_k = torch.tensor(
        np.column_stack((x_obs_k, y_obs_k)),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    U_obs = torch.tensor(
        u_obs.reshape(-1, 1),
        dtype=dtype,
        device=device,
    )

    K_obs = torch.tensor(
        k_obs.reshape(-1, 1),
        dtype=dtype,
        device=device,
    )

    X_pde = torch.tensor(
        np.column_stack((x_pde, y_pde)),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    F_pde = torch.tensor(
        f_values.reshape(-1, 1),
        dtype=dtype,
        device=device,
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.2))

        im = ax[0].imshow(
            U,
            extent=[xmin, xmax, xmin, xmax],
            origin="lower",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            alpha=0.35,
        )

        ax[0].scatter(
            x_obs,
            y_obs,
            c=u_obs,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            s=20,
            edgecolors="k",
            linewidths=0.25,
        )

        ax[0].set_title(f"$u(x,y)$ ({sampling})")
        ax[0].set_xlabel("$x$")
        ax[0].set_ylabel("$y$")
        ax[0].set_aspect("equal")

        fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

        ax[1].scatter(
            x_obs_k,
            y_obs_k,
            c=k_obs,
            cmap="viridis",
            s=20,
            edgecolors="k",
            linewidths=0.25,
        )

        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$y$")
        ax[1].set_title(r"$k$ observation locations")
        ax[1].set_aspect("equal")

        plt.tight_layout()
        plt.show()

    return (
        X_obs,
        U_obs,
        X_obs_k,
        K_obs,
        X_pde,
        F_pde,
        X,
        Y,
        U,
    )

 

def evaluate_model_inf(
    model_u,
    model_k,
    analytical_solution,
    coefficient,
    xmin=-5,
    xmax=5,
    n_grid=300,
    alpha=0.5,
    beta=5.0,
    epsilon=1.0,
    device="cpu",
    plot=False,
    verbose=False,
    save_results=False,
    results_dir="results",
):
    """
    Evaluate trained models and compare them with the exact solution.
    """

    # Evaluation grid
    x = np.linspace(xmin, xmax, n_grid)
    y = np.linspace(xmin, xmax, n_grid)

    X, Y = np.meshgrid(x, y)

    U = analytical_solution(X, Y, alpha, beta)
    K_exact = coefficient(X, Y, epsilon)

    X_test = torch.tensor(
        np.column_stack((X.ravel(), Y.ravel())),
        dtype=torch.float32,
        device=device,
    )

    # Predictions
    model_u.eval()
    model_k.eval()

    with torch.no_grad():
        U_pred = model_u(X_test).cpu().numpy().reshape(U.shape)
        K_pred = model_k(X_test).cpu().numpy().reshape(U.shape)

    # Errors
    err_u = np.linalg.norm(U_pred - U) / np.linalg.norm(U)
    err_k = np.linalg.norm(K_pred - K_exact) / np.linalg.norm(K_exact)

    if verbose:
        print(f"Relative L2 error (u): {err_u:.3e}")
        print(f"Relative L2 error (k): {err_k:.3e}")

    error_u = np.abs(U_pred - U)
    error_k = np.abs(K_pred - K_exact)

    if plot:
        fig, ax = plt.subplots(
            2, 3,
            figsize=(9, 5),
            constrained_layout=True,
        )

        extent = [xmin, xmax, xmin, xmax]

        # ---------------------- u ----------------------
        ax[0,0].imshow(U, extent=extent, origin="lower",
                    cmap="RdBu_r", vmin=-1, vmax=1)
        ax[0,0].set_title("Exact $u$")

        ax[0,1].imshow(U_pred, extent=extent, origin="lower",
                    cmap="RdBu_r", vmin=-1, vmax=1)
        ax[0,1].set_title("Predicted $u$")

        ax[0,2].imshow(error_u, extent=extent, origin="lower",
                    cmap="viridis")
        ax[0,2].set_title(r"$|u-\hat u|$")

        # ---------------------- k ----------------------
        ax[1,0].imshow(K_exact, extent=extent, origin="lower",
                    cmap="GnBu", vmin=1, vmax=3)
        ax[1,0].set_title("Exact $k$")

        ax[1,1].imshow(K_pred, extent=extent, origin="lower",
                    cmap="GnBu", vmin=1, vmax=3)
        ax[1,1].set_title("Predicted $k$")

        ax[1,2].imshow(error_k, extent=extent, origin="lower",
                    cmap="viridis")
        ax[1,2].set_title(r"$|k-\hat k|$")

        # Remove ticks
        for a in ax.flat:
            a.set_xticks([])
            a.set_yticks([])

        # Colorbars
        cbar1 = fig.colorbar(
            ax[0,1].images[0],
            ax=ax[0,:2],
            orientation="horizontal",
            fraction=0.05,
            pad=0.08,
        )
        cbar1.set_label(r"$u(x,y)$")

        cbar2 = fig.colorbar(
            ax[1,1].images[0],
            ax=ax[1,:2],
            orientation="horizontal",
            fraction=0.05,
            pad=0.08,
        )
        cbar2.set_label(r"$k(x,y)$")

        cbar3 = fig.colorbar(
            ax[0,2].images[0],
            ax=ax[:,2],
            orientation="horizontal",
            fraction=0.018,
            pad=0.12,
        )
        cbar3.set_label("Absolute error")

        plt.show()

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    if save_results:

        os.makedirs(results_dir, exist_ok=True)

        mean_error = 0.5 * (err_u + err_k)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = (
            f"infinite_problem_results_{timestamp}.txt"
        )

        filepath = os.path.join(results_dir, filename)

        with open(filepath, "w") as f:

            f.write("Infinite-domain inverse problem\n")
            f.write("=" * 40 + "\n\n")

            f.write(
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write(f"Relative L2 error (u): {err_u:.8e}\n")
            f.write(f"Relative L2 error (k): {err_k:.8e}\n")
            f.write(f"Mean error         : {mean_error:.8e}\n")

        if verbose:
            print(f"Results saved to: {filepath}")

    return err_u, err_k



