
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

    # ============================================================
    # Training domain
    # ============================================================
    train_domain=(-5.0, 5.0, -5.0, 5.0),

    # ============================================================
    # Evaluation / visualization domain
    # ============================================================
    eval_domain=(-8.0, 8.0, -8.0, 8.0),

    n_obs_u=100,
    n_obs_k=100,
    n_pde=10_000,
    n_grid=300,

    sampling="uniform",
    # "uniform"
    # "gaussian"
    # "gaussian_exponential"

    sigma=2.5,
    exp_scale=1.0,

    device="cpu",
    dtype=torch.float32,

    plot=False,
    seed=1,
):
    """
    Generate data for the infinite-domain inverse problem.

    Parameters
    ----------
    alpha : float
        Parameter of the analytical solution.

    beta : float
        Frequency parameter of the analytical solution.

    epsilon : float
        Parameter of the coefficient function.

    train_domain : tuple
        Training rectangle:
        (train_xmin, train_xmax, train_ymin, train_ymax)

    eval_domain : tuple
        Finite visualization/evaluation rectangle:
        (eval_xmin, eval_xmax, eval_ymin, eval_ymax)

    sampling : str
        Sampling strategy:

        "uniform"
            Uniform sampling inside the training rectangle.

        "gaussian"
            Untruncated Gaussian sampling in both x and y.

        "gaussian_exponential"
            Untruncated Gaussian sampling in x and
            untruncated symmetric exponential/Laplace
            sampling in y.

    sigma : float
        Standard deviation of Gaussian sampling.

    exp_scale : float
        Scale parameter of the Laplace distribution.

    Returns
    -------
    X_obs : torch.Tensor
        Observation coordinates for u.

    U_obs : torch.Tensor
        Exact observations of u.

    X_obs_k : torch.Tensor
        Observation coordinates for k.

    K_obs : torch.Tensor
        Exact observations of k.

    X_pde : torch.Tensor
        PDE collocation points.

    F_pde : torch.Tensor
        PDE source values.

    X, Y : numpy.ndarray
        Evaluation grid.

    U : numpy.ndarray
        Exact solution on evaluation grid.

    K : numpy.ndarray
        Exact coefficient on evaluation grid.
    """

    # ============================================================
    # Random number generator
    # ============================================================
    rng = np.random.default_rng(seed)

    # ============================================================
    # Training domain
    # ============================================================
    (
        train_xmin,
        train_xmax,
        train_ymin,
        train_ymax,
    ) = train_domain

    # ============================================================
    # Evaluation domain
    # ============================================================
    (
        eval_xmin,
        eval_xmax,
        eval_ymin,
        eval_ymax,
    ) = eval_domain

    # ============================================================
    # Sampling functions
    # ============================================================

    def gaussian_sampling(n):
        """
        Untruncated Gaussian sampling.

        Samples are allowed to lie anywhere in (-inf, inf).
        """
        return rng.normal(
            loc=0.0,
            scale=sigma,
            size=n,
        )

    # ------------------------------------------------------------
    # Symmetric exponential / Laplace sampling
    # ------------------------------------------------------------

    def exponential_sampling(n):
        """
        Untruncated symmetric exponential (Laplace) sampling.

        Samples are allowed to lie anywhere in (-inf, inf).
        """
        return rng.laplace(
            loc=0.0,
            scale=exp_scale,
            size=n,
        )

    # ============================================================
    # General sampling function
    # ============================================================

    def sample_points(n):

        # --------------------------------------------------------
        # Uniform sampling
        # --------------------------------------------------------
        if sampling == "uniform":

            # Uniform points ONLY inside the training rectangle
            x = rng.uniform(
                train_xmin,
                train_xmax,
                n,
            )

            y = rng.uniform(
                train_ymin,
                train_ymax,
                n,
            )

        # --------------------------------------------------------
        # Gaussian sampling
        # --------------------------------------------------------
        elif sampling == "gaussian":

            # UNTRUNCATED
            x = gaussian_sampling(n)
            y = gaussian_sampling(n)

        # --------------------------------------------------------
        # Gaussian-exponential sampling
        # --------------------------------------------------------
        elif sampling == "gaussian_exponential":

            # UNTRUNCATED
            x = gaussian_sampling(n)
            y = exponential_sampling(n)

        else:

            raise ValueError(
                f"Unknown sampling '{sampling}'. "
                "Choose from: "
                "'uniform', "
                "'gaussian', "
                "'gaussian_exponential'."
            )

        return x, y

    # ============================================================
    # Symbolic variables
    # ============================================================

    xs, ys = sp.symbols("x y")

    alpha_s, beta_s = sp.symbols(
        "alpha beta",
        positive=True,
    )

    eps_s = sp.symbols(
        "epsilon",
        positive=True,
    )

    # ============================================================
    # Observation points
    # ============================================================

    x_obs, y_obs = sample_points(
        n_obs_u
    )

    # ------------------------------------------------------------
    # k observations
    # ------------------------------------------------------------

    x_obs_k, y_obs_k = sample_points(
        n_obs_k
    )

    # ============================================================
    # Exact observations
    # ============================================================

    u_obs = analytical_solution_inf(
        x_obs,
        y_obs,
        alpha,
        beta,
    )

    k_obs = coefficient_inf(
        x_obs_k,
        y_obs_k,
        epsilon,
    )

    # ============================================================
    # PDE collocation points
    # ============================================================

    x_pde, y_pde = sample_points(
        n_pde
    )

    # ============================================================
    # PDE source term
    # ============================================================

    f_pde = source_term_inf(
        xs,
        ys,
        alpha_s,
        beta_s,
        eps_s,
    )

    f_values = f_pde(
        x_pde,
        y_pde,
        alpha,
        beta,
        epsilon,
    )

    # ============================================================
    # Evaluation / visualization grid
    #
    # IMPORTANT:
    # This grid is independent of the sampling distribution.
    #
    # It is simply a finite window used to visualize the
    # infinite-domain solution.
    # ============================================================

    x = np.linspace(
        eval_xmin,
        eval_xmax,
        n_grid,
    )

    y = np.linspace(
        eval_ymin,
        eval_ymax,
        n_grid,
    )

    X, Y = np.meshgrid(
        x,
        y,
    )

    # ============================================================
    # Exact solution on evaluation domain
    # ============================================================

    U = analytical_solution_inf(
        X,
        Y,
        alpha,
        beta,
    )

    K = coefficient_inf(
        X,
        Y,
        epsilon,
    )

    # ============================================================
    # Torch tensors
    # ============================================================

    X_obs = torch.tensor(
        np.column_stack(
            (
                x_obs,
                y_obs,
            )
        ),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    X_obs_k = torch.tensor(
        np.column_stack(
            (
                x_obs_k,
                y_obs_k,
            )
        ),
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
        np.column_stack(
            (
                x_pde,
                y_pde,
            )
        ),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    F_pde = torch.tensor(
        f_values.reshape(-1, 1),
        dtype=dtype,
        device=device,
    )

    # ============================================================
    # Plot
    # ============================================================

    if plot:

        fig, ax = plt.subplots(
            1,
            2,
            figsize=(7.0, 3.4),
        )

        # ========================================================
        # Training rectangle
        # ========================================================

        rect_u = plt.Rectangle(
            (
                train_xmin,
                train_ymin,
            ),
            train_xmax - train_xmin,
            train_ymax - train_ymin,
            fill=False,
            edgecolor="#BDBDBD",
            linewidth=1.2,
            linestyle="-",
        )

        rect_k = plt.Rectangle(
            (
                train_xmin,
                train_ymin,
            ),
            train_xmax - train_xmin,
            train_ymax - train_ymin,
            fill=False,
            edgecolor="#BDBDBD",
            linewidth=1.2,
            linestyle="-",
        )

        # ========================================================
        # u
        # ========================================================

        im_u = ax[0].imshow(
            U,
            extent=[
                eval_xmin,
                eval_xmax,
                eval_ymin,
                eval_ymax,
            ],
            origin="lower",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            alpha=0.35,
            aspect="equal",
        )

        # --------------------------------------------------------
        # u observation points
        # --------------------------------------------------------

        ax[0].scatter(
            x_obs,
            y_obs,
            c=u_obs,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            s=20,
            edgecolors="#BDBDBD",
            linewidths=0.45,
        )

        # --------------------------------------------------------
        # Training rectangle
        # --------------------------------------------------------

        ax[0].add_patch(
            rect_u
        )

        # --------------------------------------------------------
        # Limits
        # --------------------------------------------------------

        ax[0].set_xlim(
            eval_xmin,
            eval_xmax,
        )

        ax[0].set_ylim(
            eval_ymin,
            eval_ymax,
        )

        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------

        ax[0].set_title(
            f"$u(x,y)$ ({sampling})"
        )

        ax[0].set_xlabel("$x$")
        ax[0].set_ylabel("$y$")

        ax[0].set_aspect(
            "equal"
        )

        # --------------------------------------------------------
        # Colorbar
        # --------------------------------------------------------

        cbar_u = fig.colorbar(
            im_u,
            ax=ax[0],
            fraction=0.046,
            pad=0.04,
        )

        cbar_u.set_label(
            r"$u$",
            fontsize=8,
        )

        cbar_u.ax.tick_params(
            labelsize=7,
        )

        # ========================================================
        # k
        # ========================================================

        im_k = ax[1].imshow(
            K,
            extent=[
                eval_xmin,
                eval_xmax,
                eval_ymin,
                eval_ymax,
            ],
            origin="lower",
            cmap="GnBu",
            vmin=1,
            vmax=3,
            alpha=0.35,
            aspect="equal",
        )

        # --------------------------------------------------------
        # k observation points
        # --------------------------------------------------------

        ax[1].scatter(
            x_obs_k,
            y_obs_k,
            c=k_obs,
            cmap="GnBu",
            vmin=1,
            vmax=3,
            s=20,
            edgecolors="#BDBDBD",
            linewidths=0.45,
        )

        # --------------------------------------------------------
        # Training rectangle
        # --------------------------------------------------------

        ax[1].add_patch(
            rect_k
        )

        # --------------------------------------------------------
        # Limits
        # --------------------------------------------------------

        ax[1].set_xlim(
            eval_xmin,
            eval_xmax,
        )

        ax[1].set_ylim(
            eval_ymin,
            eval_ymax,
        )

        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------

        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$y$")

        ax[1].set_title(
            r"$k(x,y)$"
        )

        ax[1].set_aspect(
            "equal"
        )

        # --------------------------------------------------------
        # Colorbar
        # --------------------------------------------------------

        cbar_k = fig.colorbar(
            im_k,
            ax=ax[1],
            fraction=0.046,
            pad=0.04,
        )

        cbar_k.set_label(
            r"$k$",
            fontsize=8,
        )

        cbar_k.ax.tick_params(
            labelsize=7,
        )

        # ========================================================
        # Common ticks
        # ========================================================

        ticks_x = [
            eval_xmin,
            train_xmin,
            0.0,
            train_xmax,
            eval_xmax,
        ]

        ticks_y = [
            eval_ymin,
            train_ymin,
            0.0,
            train_ymax,
            eval_ymax,
        ]

        for a in ax:

            a.set_xticks(
                ticks_x
            )

            a.set_yticks(
                ticks_y
            )

            a.tick_params(
                labelsize=8,
                length=3,
            )

        # ========================================================
        # Overall title
        # ========================================================

        fig.suptitle(
            sampling.capitalize(),
            fontsize=9,
            y=1.02,
        )

        plt.tight_layout()

        plt.show()

    # ============================================================
    # Return
    # ============================================================

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
        K,
    )


# def generate_dataset_inf(
#     alpha_true=0.5,
#     alpha_sampling=None,
#     beta=5.0,
#     epsilon=1.0,
#     domain=(-5.0, 5.0),
#     n_obs_u=100,
#     n_obs_k=100,
#     n_pde=10_000,
#     n_grid=300,
#     sampling="uniform",   # "uniform", "gaussian", "gaussian_exponential"
#     sigma=None,
#     sampling_scale=1.0,
#     exp_scale=1.0,
#     device="cpu",
#     dtype=torch.float32,
#     plot=False,
#     seed=1
# ):

#     rng = np.random.default_rng(seed)

#     xmin, xmax = domain

#     # --------------------------------------------------
#     # Sampling scale
#     # --------------------------------------------------

#     if sampling in ["gaussian", "gaussian_exponential"]:

#         if alpha_sampling is None:
#             raise ValueError(
#                 "alpha_sampling must be provided for "
#                 "Gaussian-based sampling."
#             )

#         # Adaptive Gaussian width
#         #
#         # sigma = sampling_scale / alpha_sampling
#         #
#         sigma = sampling_scale / alpha_sampling

#     elif sigma is None:

#         # sigma is not used for uniform sampling
#         sigma = 1.0

#     # --------------------------------------------------
#     # Sampling functions
#     # --------------------------------------------------

#     def truncated_gaussian(n):

#         x = rng.normal(
#             0.0,
#             sigma,
#             5 * n
#         )

#         x = x[
#             (x >= xmin) &
#             (x <= xmax)
#         ]

#         while len(x) < n:

#             extra = rng.normal(
#                 0.0,
#                 sigma,
#                 2 * n
#             )

#             extra = extra[
#                 (extra >= xmin) &
#                 (extra <= xmax)
#             ]

#             x = np.concatenate(
#                 (x, extra)
#             )

#         return x[:n]

#     def truncated_exponential(n):

#         # Symmetric exponential / Laplace

#         y = rng.laplace(
#             0.0,
#             exp_scale,
#             3 * n
#         )

#         y = y[
#             (y >= xmin) &
#             (y <= xmax)
#         ]

#         while len(y) < n:

#             extra = rng.laplace(
#                 0.0,
#                 exp_scale,
#                 2 * n
#             )

#             extra = extra[
#                 (extra >= xmin) &
#                 (extra <= xmax)
#             ]

#             y = np.concatenate(
#                 (y, extra)
#             )

#         return y[:n]

#     def sample_points(n):

#         if sampling == "uniform":

#             x = rng.uniform(
#                 xmin,
#                 xmax,
#                 n
#             )

#             y = rng.uniform(
#                 xmin,
#                 xmax,
#                 n
#             )

#         elif sampling == "gaussian":

#             x = truncated_gaussian(n)
#             y = truncated_gaussian(n)

#         elif sampling == "gaussian_exponential":

#             x = truncated_gaussian(n)
#             y = truncated_exponential(n)

#         else:

#             raise ValueError(
#                 f"Unknown sampling '{sampling}'"
#             )

#         return x, y

#     # --------------------------------------------------
#     # Symbolic variables
#     # --------------------------------------------------

#     xs, ys = sp.symbols("x y")

#     alpha_s, beta_s = sp.symbols(
#         "alpha beta",
#         positive=True
#     )

#     eps_s = sp.symbols(
#         "epsilon",
#         positive=True
#     )

#     # --------------------------------------------------
#     # Observation points
#     # --------------------------------------------------

#     x_obs, y_obs = sample_points(n_obs_u)

#     # k observations
#     x_obs_k, y_obs_k = sample_points(n_obs_k)

#     # --------------------------------------------------
#     # TRUE manufactured solution
#     #
#     # IMPORTANT:
#     # alpha_true is used here, NOT alpha_sampling.
#     # --------------------------------------------------

#     u_obs = analytical_solution_inf(
#         x_obs,
#         y_obs,
#         alpha_true,
#         beta
#     )

#     k_obs = coefficient_inf(
#         x_obs_k,
#         y_obs_k,
#         epsilon
#     )

#     # --------------------------------------------------
#     # PDE collocation points
#     # --------------------------------------------------

#     x_pde, y_pde = sample_points(n_pde)

#     # --------------------------------------------------
#     # TRUE PDE source
#     #
#     # Again, alpha_true is used here.
#     # --------------------------------------------------

#     f_pde = source_term_inf(
#         xs,
#         ys,
#         alpha_s,
#         beta_s,
#         eps_s
#     )

#     f_values = f_pde(
#         x_pde,
#         y_pde,
#         alpha_true,
#         beta,
#         epsilon
#     )

#     # --------------------------------------------------
#     # Visualization grid
#     # --------------------------------------------------

#     x = np.linspace(
#         xmin,
#         xmax,
#         n_grid
#     )

#     y = np.linspace(
#         xmin,
#         xmax,
#         n_grid
#     )

#     X, Y = np.meshgrid(x, y)

#     U = analytical_solution_inf(
#         X,
#         Y,
#         alpha_true,
#         beta
#     )

#     K = coefficient_inf(
#         X,
#         Y,
#         epsilon
#     )

#     # --------------------------------------------------
#     # Torch tensors
#     # --------------------------------------------------

#     X_obs = torch.tensor(
#         np.column_stack(
#             (x_obs, y_obs)
#         ),
#         dtype=dtype,
#         device=device,
#         requires_grad=True,
#     )

#     X_obs_k = torch.tensor(
#         np.column_stack(
#             (x_obs_k, y_obs_k)
#         ),
#         dtype=dtype,
#         device=device,
#         requires_grad=True,
#     )

#     U_obs = torch.tensor(
#         u_obs.reshape(-1, 1),
#         dtype=dtype,
#         device=device,
#     )

#     K_obs = torch.tensor(
#         k_obs.reshape(-1, 1),
#         dtype=dtype,
#         device=device,
#     )

#     X_pde = torch.tensor(
#         np.column_stack(
#             (x_pde, y_pde)
#         ),
#         dtype=dtype,
#         device=device,
#         requires_grad=True,
#     )

#     F_pde = torch.tensor(
#         f_values.reshape(-1, 1),
#         dtype=dtype,
#         device=device,
#     )

#     # --------------------------------------------------
#     # Plot
#     # --------------------------------------------------

#     if plot:

#         fig, ax = plt.subplots(
#             1,
#             2,
#             figsize=(6.5, 3.2)
#         )

#         # ----------------------------------------------
#         # u
#         # ----------------------------------------------

#         im = ax[0].imshow(
#             U,
#             extent=[
#                 xmin,
#                 xmax,
#                 xmin,
#                 xmax
#             ],
#             origin="lower",
#             cmap="RdBu_r",
#             vmin=-1,
#             vmax=1,
#             alpha=0.35,
#         )

#         ax[0].scatter(
#             x_obs,
#             y_obs,
#             c=u_obs,
#             cmap="RdBu_r",
#             vmin=-1,
#             vmax=1,
#             s=20,
#             edgecolors="k",
#             linewidths=0.25,
#         )

#         #if sampling in [
#         #    "gaussian",
#         #    "gaussian_exponential"
#         #]:

#             #ax[0].set_title(
#             #    rf"$u(x,y)$ ({sampling}, "
#                 #rf"$\sigma={sigma:.2f}$)"
#             #)

#         #else:

#          #   ax[0].set_title(
#          #       rf"$u(x,y)$ ({sampling})"
#          #   )

#         ax[0].set_xlabel("$x$")
#         ax[0].set_ylabel("$y$")
#         ax[0].set_aspect("equal")

#         fig.colorbar(
#             im,
#             ax=ax[0],
#             fraction=0.046,
#             pad=0.04
#         )

#         # ----------------------------------------------
#         # k
#         # ----------------------------------------------

#         im = ax[1].imshow(
#             K,
#             extent=[
#                 xmin,
#                 xmax,
#                 xmin,
#                 xmax
#             ],
#             origin="lower",
#             vmin=1,
#             vmax=3,
#             alpha=0.35,
#         )

#         ax[1].scatter(
#             x_obs_k,
#             y_obs_k,
#             c=k_obs,
#             cmap="viridis",
#             s=20,
#             edgecolors="k",
#             linewidths=0.25,
#         )

#         ax[1].set_xlabel("$x$")
#         ax[1].set_ylabel("$y$")
#         #ax[1].set_title(
#         #    r"$k$ observation locations"
#         #)
#         ax[1].set_aspect("equal")

#         fig.colorbar(
#             im,
#             ax=ax[1],
#             fraction=0.046,
#             pad=0.04
#         )

#         plt.tight_layout()
#         plt.show()

#     # --------------------------------------------------
#     # Return
#     # --------------------------------------------------

#     return (
#         X_obs,
#         U_obs,
#         X_obs_k,
#         K_obs,
#         X_pde,
#         F_pde,
#         X,
#         Y,
#         U,
#     )
 

# 




def evaluate_model_inf(
    model_u,
    model_k,
    analytical_solution,
    coefficient,
    X_obs=None,
    X_obs_k=None,
    sampling="uniform",

    # ----------------------------------------------------------
    # Training domain
    # ----------------------------------------------------------

    train_xmin=-5.0,
    train_xmax=5.0,
    train_ymin=-5.0,
    train_ymax=5.0,

    # ----------------------------------------------------------
    # Evaluation domain
    # ----------------------------------------------------------

    eval_xmin=-8.0,
    eval_xmax=8.0,
    eval_ymin=-8.0,
    eval_ymax=8.0,

    n_grid=400,

    alpha=0.5,
    beta=5.0,
    epsilon=1.0,

    device="cpu",

    plot=False,
    show_bar_ylabel=False,
    show_legend=False,
    verbose=False,

    save_results=False,
    results_dir="results",

    save_plot=False,
    plot_path=None,
):

    """
    Evaluate spatial generalization outside the training domain.

    Figure structure:

                    Prediction          Absolute error
                ┌──────────────────┬──────────────────┐
            u   │      u_hat       │    |u-u_hat|     │
                ├──────────────────┼──────────────────┤
            k   │      k_hat       │    |k-k_hat|     │
                ├──────────────────┴──────────────────┤
                │        Relative L2 errors            │
                │  Inside | Outside | Global          │
                └─────────────────────────────────────┘

    The training-domain rectangle is overlaid on the
    prediction and error maps.

    Errors are reported globally, inside the training
    domain, and outside the training domain.

    Returns
    -------
    dict
        Dictionary containing relative L2 errors,
        predictions, pointwise errors, and evaluation grid.
    """

    # ============================================================
    # Evaluation grid
    # ============================================================

    x = np.linspace(
        eval_xmin,
        eval_xmax,
        n_grid,
    )

    y = np.linspace(
        eval_ymin,
        eval_ymax,
        n_grid,
    )

    X, Y = np.meshgrid(x, y)

    # ============================================================
    # Exact solutions on the larger evaluation domain
    # ============================================================

    U = analytical_solution(
        X,
        Y,
        alpha,
        beta,
    )

    K_exact = coefficient(
        X,
        Y,
        epsilon,
    )

    # ============================================================
    # Test points
    # ============================================================

    X_test = torch.tensor(
        np.column_stack(
            (
                X.ravel(),
                Y.ravel(),
            )
        ),
        dtype=torch.float32,
        device=device,
    )

    # ============================================================
    # Predictions
    # ============================================================

    model_u.eval()
    model_k.eval()

    with torch.no_grad():

        U_pred = (
            model_u(X_test)
            .cpu()
            .numpy()
            .reshape(U.shape)
        )

        K_pred = (
            model_k(X_test)
            .cpu()
            .numpy()
            .reshape(K_exact.shape)
        )

    # ============================================================
    # Pointwise errors
    # ============================================================

    error_u = np.abs(
        U_pred - U
    )

    error_k = np.abs(
        K_pred - K_exact
    )

    # ============================================================
    # Global relative L2 errors
    # ============================================================

    err_u_global = (
        np.linalg.norm(
            U_pred - U
        )
        /
        (
            np.linalg.norm(U)
            + 1e-14
        )
    )

    err_k_global = (
        np.linalg.norm(
            K_pred - K_exact
        )
        /
        (
            np.linalg.norm(K_exact)
            + 1e-14
        )
    )

    # ============================================================
    # Outside-training-domain mask
    #
    # True = points outside
    # [train_xmin, train_xmax] x
    # [train_ymin, train_ymax]
    # ============================================================

    outside_mask = (
        (X < train_xmin)
        |
        (X > train_xmax)
        |
        (Y < train_ymin)
        |
        (Y > train_ymax)
    )

    # ============================================================
    # Outside-domain errors
    # ============================================================

    U_outside = U[outside_mask]
    U_pred_outside = U_pred[outside_mask]

    K_outside = K_exact[outside_mask]
    K_pred_outside = K_pred[outside_mask]

    err_u_outside = (
        np.linalg.norm(
            U_pred_outside - U_outside
        )
        /
        (
            np.linalg.norm(U_outside)
            + 1e-14
        )
    )

    err_k_outside = (
        np.linalg.norm(
            K_pred_outside - K_outside
        )
        /
        (
            np.linalg.norm(K_outside)
            + 1e-14
        )
    )

    # ============================================================
    # Inside-domain errors
    # ============================================================

    inside_mask = ~outside_mask

    err_u_inside = (
        np.linalg.norm(
            U_pred[inside_mask]
            -
            U[inside_mask]
        )
        /
        (
            np.linalg.norm(
                U[inside_mask]
            )
            + 1e-14
        )
    )

    err_k_inside = (
        np.linalg.norm(
            K_pred[inside_mask]
            -
            K_exact[inside_mask]
        )
        /
        (
            np.linalg.norm(
                K_exact[inside_mask]
            )
            + 1e-14
        )
    )

    # ============================================================
    # Verbose
    # ============================================================

    if verbose:

        print("\n========================================")
        print("Spatial generalization")
        print("========================================")

        print(
            f"Training domain : "
            f"[{train_xmin}, {train_xmax}] × "
            f"[{train_ymin}, {train_ymax}]"
        )

        print(
            f"Evaluation domain : "
            f"[{eval_xmin}, {eval_xmax}] × "
            f"[{eval_ymin}, {eval_ymax}]"
        )

        print("\nGlobal relative L2 error")

        print(
            f"u : {err_u_global:.3e}"
        )

        print(
            f"k : {err_k_global:.3e}"
        )

        print("\nInside training domain")

        print(
            f"u : {err_u_inside:.3e}"
        )

        print(
            f"k : {err_k_inside:.3e}"
        )

        print("\nOutside training domain")

        print(
            f"u : {err_u_outside:.3e}"
        )

        print(
            f"k : {err_k_outside:.3e}"
        )

    # ============================================================
    # Convert sampling points
    # ============================================================

    def to_numpy(points):

        if points is None:
            return None

        if torch.is_tensor(points):

            return (
                points
                .detach()
                .cpu()
                .numpy()
            )

        return np.asarray(points)

    X_obs_np = to_numpy(X_obs)
    X_obs_k_np = to_numpy(X_obs_k)

    # ============================================================
    # Plot
    # ============================================================

    if plot:

        # --------------------------------------------------------
        # Figure and GridSpec
        # --------------------------------------------------------

        fig = plt.figure(
            figsize=(3.5, 7.0)
        )

        gs = fig.add_gridspec(
            3,
            2,

            # Third row slightly shorter
            height_ratios=[
                1.0,
                1.0,
                0.5,
            ],

            hspace=0.45,
            wspace=0.50,
        )

        # --------------------------------------------------------
        # Spatial panels
        # --------------------------------------------------------

        ax00 = fig.add_subplot(
            gs[0, 0]
        )

        ax01 = fig.add_subplot(
            gs[0, 1]
        )

        ax10 = fig.add_subplot(
            gs[1, 0]
        )

        ax11 = fig.add_subplot(
            gs[1, 1]
        )

        # --------------------------------------------------------
        # Third row spans both columns
        # --------------------------------------------------------

        ax_error = fig.add_subplot(
            gs[2, :]
        )

        # ========================================================
        # Common extent
        # ========================================================

        extent = [
            eval_xmin,
            eval_xmax,
            eval_ymin,
            eval_ymax,
        ]

        # ========================================================
        # Fixed solution scales
        # ========================================================

        u_vmin = -1
        u_vmax = 1

        k_vmin = 1
        k_vmax = 3

        # ========================================================
        # Error scales
        # ========================================================

        err_u_max = 0.4#np.max(error_u)

        err_k_max = 0.4#np.max(error_k)

        if err_u_max == 0:
            err_u_max = 1e-12

        if err_k_max == 0:
            err_k_max = 1e-12

        # ========================================================
        # ROW 1 — u prediction
        # ========================================================

        im_u = ax00.imshow(
            U_pred,
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            vmin=u_vmin,
            vmax=u_vmax,
            aspect="equal",
        )

        # --------------------------------------------------------
        # u sampling
        # --------------------------------------------------------

        if X_obs_np is not None:

            ax00.scatter(
                X_obs_np[:, 0],
                X_obs_np[:, 1],

                facecolors="none",
                edgecolors="#BDBDBD",

                s=20,
                alpha=0.8,
                linewidths=0.6,
            )

        # --------------------------------------------------------
        # Training-domain rectangle
        # --------------------------------------------------------

        rect_u = plt.Rectangle(
            (
                train_xmin,
                train_ymin,
            ),

            train_xmax - train_xmin,
            train_ymax - train_ymin,

            fill=False,
            edgecolor="#BDBDBD",
            linewidth=1.0,
            linestyle="-",
        )

        ax00.add_patch(
            rect_u
        )

        # ========================================================
        # ROW 1 — u error
        # ========================================================

        im_err_u = ax01.imshow(
            error_u,
            extent=extent,
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=err_u_max,
            aspect="equal",
        )

        # --------------------------------------------------------
        # u sampling
        # --------------------------------------------------------

        if X_obs_np is not None:

            ax01.scatter(
                X_obs_np[:, 0],
                X_obs_np[:, 1],

                facecolors="none",
                edgecolors="#BDBDBD",

                s=20,
                alpha=0.8,
                linewidths=0.6,
            )

        # --------------------------------------------------------
        # Training-domain rectangle
        # --------------------------------------------------------

        rect_u_err = plt.Rectangle(
            (
                train_xmin,
                train_ymin,
            ),

            train_xmax - train_xmin,
            train_ymax - train_ymin,

            fill=False,
            edgecolor="#BDBDBD",
            linewidth=1.0,
            linestyle="-",
        )

        ax01.add_patch(
            rect_u_err
        )

        # ========================================================
        # ROW 2 — k prediction
        # ========================================================

        im_k = ax10.imshow(
            K_pred,
            extent=extent,
            origin="lower",
            cmap="GnBu",
            vmin=k_vmin,
            vmax=k_vmax,
            aspect="equal",
        )

        # --------------------------------------------------------
        # k sampling
        # --------------------------------------------------------

        if X_obs_k_np is not None:

            ax10.scatter(
                X_obs_k_np[:, 0],
                X_obs_k_np[:, 1],

                facecolors="none",
                edgecolors="#BDBDBD",

                s=20,
                alpha=0.8,
                linewidths=0.6,
            )

        # --------------------------------------------------------
        # Training-domain rectangle
        # --------------------------------------------------------

        rect_k = plt.Rectangle(
            (
                train_xmin,
                train_ymin,
            ),

            train_xmax - train_xmin,
            train_ymax - train_ymin,

            fill=False,
            edgecolor="#BDBDBD",
            linewidth=1.0,
            linestyle="-",
        )

        ax10.add_patch(
            rect_k
        )

        ax00.set_xlim(eval_xmin, eval_xmax)
        ax00.set_ylim(eval_ymin, eval_ymax)

        ax01.set_xlim(eval_xmin, eval_xmax)
        ax01.set_ylim(eval_ymin, eval_ymax)

        # ========================================================
        # ROW 2 — k error
        # ========================================================

        im_err_k = ax11.imshow(
            error_k,
            extent=extent,
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=err_k_max,
            aspect="equal",
        )

        # --------------------------------------------------------
        # k sampling
        # --------------------------------------------------------

        if X_obs_k_np is not None:

            ax11.scatter(
                X_obs_k_np[:, 0],
                X_obs_k_np[:, 1],

                facecolors="none",
                edgecolors="#BDBDBD",

                s=20,
                alpha=0.8,
                linewidths=0.6,
            )

        # --------------------------------------------------------
        # Training-domain rectangle
        # --------------------------------------------------------

        rect_k_err = plt.Rectangle(
            (
                train_xmin,
                train_ymin,
            ),

            train_xmax - train_xmin,
            train_ymax - train_ymin,

            fill=False,
            edgecolor="#BDBDBD",
            linewidth=1.0,
            linestyle="-",
        )

        ax11.add_patch(
            rect_k_err
        )
        ax10.set_xlim(eval_xmin, eval_xmax)
        ax10.set_ylim(eval_ymin, eval_ymax)
        ax11.set_xlim(eval_xmin, eval_xmax)
        ax11.set_ylim(eval_ymin, eval_ymax)
        # ========================================================
        # ROW 3 — Relative L2 errors
        # ========================================================

        regions = [
            "Inside",
            "Outside",
        ]

        u_errors = [
            err_u_inside,
            err_u_outside,
        ]

        k_errors = [
            err_k_inside,
            err_k_outside,
        ]

        x_pos = np.arange(
            len(regions)
        )

        width = 0.40

        u_color = "#2255a080"   # blue
        k_color = "#91cc9b"   # gray

        # --------------------------------------------------------
        # u bars
        # --------------------------------------------------------

        bars_u = ax_error.bar(
            x_pos - width / 2,
            u_errors,
            color=u_color,
            width=width,
            label=r"$u$",
        )

        # --------------------------------------------------------
        # k bars
        # --------------------------------------------------------

        bars_k = ax_error.bar(
            x_pos + width / 2,
            k_errors,
            color=k_color,
            width=width,
            label=r"$k$",
        )

        # --------------------------------------------------------
        # Logarithmic scale
        # --------------------------------------------------------

        ax_error.set_yscale(
            "log"
        )
 
        ax_error.set_ylim(1e-4, 1e7)
 
        # --------------------------------------------------------
        # X axis
        # --------------------------------------------------------

        ax_error.set_xticks(
            x_pos
        )

        ax_error.set_xticklabels(
            regions,
            fontsize=7,
            color="black"
        )

        # --------------------------------------------------------
        # Legend
        # --------------------------------------------------------

        if show_legend:
            ax_error.legend(
                fontsize=7,
                frameon=False,
                loc="upper right",
            )
        else:
            ax_error.legend().set_visible(False)    

        # --------------------------------------------------------
        # Tick parameters
        # --------------------------------------------------------
        axis_color = "gray"
        ax_error.tick_params(
            axis="both",
            #labelcolor="black",
            labelsize=6,
            colors=axis_color,
            length=2,
            width=0.5,
        )

        ax_error.set_xticklabels(
        regions,
        fontsize=6,
        color="black",
    )

        # --------------------------------------------------------
        # Remove top and right borders
        # --------------------------------------------------------

        ax_error.spines["top"].set_visible(False)
        ax_error.spines["right"].set_visible(False)

        ax_error.spines["left"].set_color(axis_color)
        ax_error.spines["bottom"].set_color(axis_color)

        ax_error.spines["left"].set_linewidth(0.5)
        ax_error.spines["bottom"].set_linewidth(0.5)

        # --------------------------------------------------------
        # Annotate u bars
        # --------------------------------------------------------

        for bar, value in zip(
            bars_u,
            u_errors,
        ):
            ax_error.text(
                bar.get_x()
                + bar.get_width() / 2,
                value,
                f"{value:.2e}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="gray",
                rotation=0,
            )

        # --------------------------------------------------------
        # Annotate k bars
        # --------------------------------------------------------

        for bar, value in zip(
            bars_k,
            k_errors,
        ):
            ax_error.text(
                bar.get_x()
                + bar.get_width() / 2,
                value,
                f"{value:.2e}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="gray",
                rotation=0,
            )


        # --------------------------------------------------------
        # Y-axis label
        # --------------------------------------------------------

        if show_bar_ylabel:
            ax_error.set_ylabel(
                r"Relative $L_2$ error",
                fontsize=8,
            )
        else:
            ax_error.set_ylabel("")

        # ========================================================
        # Spatial ticks and axes style
        # ========================================================
        ticks = [
            eval_xmin,
            train_xmin,
            train_xmax,
            eval_xmax,
        ]

        axis_color = "gray"

        for a in [
            ax00,
            ax01,
            ax10,
            ax11,
        ]:
            a.set_xticks(ticks)
            a.set_yticks(ticks)

            # Tick marks and tick numbers
            a.tick_params(
                axis="both",
                labelsize=6,
                colors=axis_color,
                length=2,
                width=0.5,
            )

            # Axis lines
            for spine in a.spines.values():
                spine.set_color(axis_color)
                spine.set_linewidth(0.5)

            a.set_xlabel("")
            a.set_ylabel("")

        # ========================================================
        # Sampling label
        # ========================================================

        fig.suptitle(
            sampling.capitalize(),
            fontsize=9,
            y=0.90,
        )

        # ========================================================
        # Colorbars
        # ========================================================

        cbar_u = fig.colorbar(
            im_u,
            ax=ax00,
            orientation="horizontal",
            fraction=0.028,
            pad=0.20,
            aspect=30,
        )

        cbar_u.set_label(
            r"$\hat{u}$",
            fontsize=8,
        )

        cbar_u.ax.tick_params(
            labelsize=7
        )

        # --------------------------------------------------------
        # u error colorbar
        # --------------------------------------------------------

        cbar_err_u = fig.colorbar(
            im_err_u,
            ax=ax01,
            orientation="horizontal",
            fraction=0.028,
            pad=0.20,
            aspect=30,
        )

        cbar_err_u.set_label(
            r"$|\hat{u}-u|$",
            fontsize=8,
        )

        cbar_err_u.ax.tick_params(
            labelsize=7
        )

        # --------------------------------------------------------
        # k colorbar
        # --------------------------------------------------------

        cbar_k = fig.colorbar(
            im_k,
            ax=ax10,
            orientation="horizontal",
            fraction=0.028,
            pad=0.20,
            aspect=30,
        )

        cbar_k.set_label(
            r"$\hat{k}$",
            fontsize=8,
        )

        cbar_k.ax.tick_params(
            labelsize=7
        )

        # --------------------------------------------------------
        # k error colorbar
        # --------------------------------------------------------

        cbar_err_k = fig.colorbar(
            im_err_k,
            ax=ax11,
            orientation="horizontal",
            fraction=0.028,
            pad=0.20,
            aspect=30,
        )

        cbar_err_k.set_label(
            r"$|\hat{k}-k|$",
            fontsize=8,
        )

        cbar_err_k.ax.tick_params(
            labelsize=7
        )

        for cbar in [
            cbar_u,
            cbar_err_u,
            cbar_k,
            cbar_err_k,
        ]:
            cbar.ax.tick_params(
                axis="x",
                labelsize=6,
                colors=axis_color,
                length=2,
                width=0.5,
            )

            cbar.outline.set_edgecolor(axis_color)
            cbar.outline.set_linewidth(0.5)

        # ========================================================
        # Save plot
        # ========================================================

        if save_plot:

            if plot_path is None:

                os.makedirs(
                    results_dir,
                    exist_ok=True,
                )

                timestamp = (
                    datetime.now()
                    .strftime(
                        "%Y%m%d_%H%M%S"
                    )
                )

                plot_path = os.path.join(
                    results_dir,

                    f"infinite_prediction_"
                    f"{sampling}_"
                    f"{timestamp}.png",
                )

            else:

                directory = os.path.dirname(
                    plot_path
                )

                if directory:

                    os.makedirs(
                        directory,
                        exist_ok=True,
                    )

            fig.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight",
            )

            if verbose:

                print(
                    f"Prediction plot saved to: "
                    f"{plot_path}"
                )

        plt.show()

    # ============================================================
    # Save numerical results
    # ============================================================

    if save_results:

        os.makedirs(
            results_dir,
            exist_ok=True,
        )

        mean_global_error = 0.5 * (
            err_u_global
            +
            err_k_global
        )

        mean_outside_error = 0.5 * (
            err_u_outside
            +
            err_k_outside
        )

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        filename = (
            f"infinite_problem_results_"
            f"{sampling}_"
            f"{timestamp}.txt"
        )

        filepath = os.path.join(
            results_dir,
            filename,
        )

        with open(
            filepath,
            "w",
        ) as f:

            f.write(
                "Infinite-domain inverse problem\n"
            )

            f.write(
                "=" * 50
                + "\n\n"
            )

            f.write(
                f"Date: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write(
                f"Sampling: {sampling}\n\n"
            )

            # ----------------------------------------------------
            # Training domain
            # ----------------------------------------------------

            f.write(
                "Training domain\n"
            )

            f.write(
                f"x: [{train_xmin}, {train_xmax}]\n"
            )

            f.write(
                f"y: [{train_ymin}, {train_ymax}]\n\n"
            )

            # ----------------------------------------------------
            # Evaluation domain
            # ----------------------------------------------------

            f.write(
                "Evaluation domain\n"
            )

            f.write(
                f"x: [{eval_xmin}, {eval_xmax}]\n"
            )

            f.write(
                f"y: [{eval_ymin}, {eval_ymax}]\n\n"
            )

            # ----------------------------------------------------
            # Global errors
            # ----------------------------------------------------

            f.write(
                "Global relative L2 errors\n"
            )

            f.write(
                f"u: {err_u_global:.8e}\n"
            )

            f.write(
                f"k: {err_k_global:.8e}\n"
            )

            f.write(
                f"Mean: {mean_global_error:.8e}\n\n"
            )

            # ----------------------------------------------------
            # Inside errors
            # ----------------------------------------------------

            f.write(
                "Inside training domain\n"
            )

            f.write(
                f"u: {err_u_inside:.8e}\n"
            )

            f.write(
                f"k: {err_k_inside:.8e}\n\n"
            )

            # ----------------------------------------------------
            # Outside errors
            # ----------------------------------------------------

            f.write(
                "Outside training domain\n"
            )

            f.write(
                f"u: {err_u_outside:.8e}\n"
            )

            f.write(
                f"k: {err_k_outside:.8e}\n"
            )

            f.write(
                f"Mean: {mean_outside_error:.8e}\n"
            )

        if verbose:

            print(
                f"Results saved to: "
                f"{filepath}"
            )

    # ============================================================
    # Return
    # ============================================================

    return {

        "err_u_global": err_u_global,
        "err_k_global": err_k_global,

        "err_u_inside": err_u_inside,
        "err_k_inside": err_k_inside,

        "err_u_outside": err_u_outside,
        "err_k_outside": err_k_outside,

        "U_pred": U_pred,
        "K_pred": K_pred,

        "error_u": error_u,
        "error_k": error_k,

        "X": X,
        "Y": Y,

        "outside_mask": outside_mask,
    }


def compute_training_errors(
    model_u,
    model_k,
    X_obs,
    X_obs_k,
    analytical_solution,
    coefficient,
    alpha=0.5,
    beta=5.0,
    epsilon=1.0,
    device="gpu",
):
    """
    Compute relative L2 errors of u and k at the training points.

    Returns
    -------
    err_u : float
        Relative L2 error for u.
    err_k : float
        Relative L2 error for k.
    """

    # ------------------------------------------------------------
    # Convert points to tensors
    # ------------------------------------------------------------

    if not torch.is_tensor(X_obs):
        X_obs = torch.tensor(
            X_obs,
            dtype=torch.float32,
            device=device,
        )
    else:
        X_obs = X_obs.to(device=device, dtype=torch.float32)

    if not torch.is_tensor(X_obs_k):
        X_obs_k = torch.tensor(
            X_obs_k,
            dtype=torch.float32,
            device=device,
        )
    else:
        X_obs_k = X_obs_k.to(device=device, dtype=torch.float32)

    # ------------------------------------------------------------
    # Exact values at training points
    # ------------------------------------------------------------

    X_u_np = X_obs.detach().cpu().numpy()

    u_exact = analytical_solution(
        X_u_np[:, 0],
        X_u_np[:, 1],
        alpha,
        beta,
    )

    X_k_np = X_obs_k.detach().cpu().numpy()

    k_exact = coefficient(
        X_k_np[:, 0],
        X_k_np[:, 1],
        epsilon,
    )

    # Convert exact values to tensors
    u_exact = torch.tensor(
        u_exact,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)

    k_exact = torch.tensor(
        k_exact,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    was_training_u = model_u.training
    was_training_k = model_k.training

    model_u.eval()
    model_k.eval()

    with torch.no_grad():

        u_pred = model_u(X_obs)

        k_pred = model_k(X_obs_k)

    # Restore original training state
    if was_training_u:
        model_u.train()

    if was_training_k:
        model_k.train()

    # ------------------------------------------------------------
    # Relative L2 errors
    # ------------------------------------------------------------

    err_u = (
        torch.linalg.norm(u_pred - u_exact)
        / (torch.linalg.norm(u_exact) + 1e-14)
    ).item()

    err_k = (
        torch.linalg.norm(k_pred - k_exact)
        / (torch.linalg.norm(k_exact) + 1e-14)
    ).item()

    return err_u, err_k