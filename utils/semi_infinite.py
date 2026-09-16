import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch


def source_term_semi_inf(xs_semi, ys_semi, alpha_s_semi, beta_s_semi, eps_s_semi):
    """Return a numerically stable NumPy-callable Poisson source."""
    del xs_semi, ys_semi, alpha_s_semi, beta_s_semi, eps_s_semi

    def source(x, y, alpha, beta, epsilon):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.clip((y + 1.5) / epsilon, -60.0, 60.0)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        coefficient = 1.0 + 2.0 * sigmoid
        coefficient_y = 2.0 * sigmoid * (1.0 - sigmoid) / epsilon
        envelope = np.exp(-alpha * (x**2 + y**2))
        laplacian_factor = (
            (4.0 * alpha**2 * (x**2 + y**2) - 4.0 * alpha - beta**2)
            * np.cos(beta * x)
            + 4.0 * alpha * beta * x * np.sin(beta * x)
        )
        return envelope * (
            -coefficient * laplacian_factor
            + 2.0 * alpha * y * coefficient_y * np.cos(beta * x)
        )

    return source


def analytical_solution_semi_inf(xs_semi, ys_semi, alpha_s_semi, beta_s_semi):
    """Evaluate the semi-infinite manufactured solution."""
    return np.exp(
        -alpha_s_semi * (xs_semi**2 + ys_semi**2)
    ) * np.cos(beta_s_semi * xs_semi)


def coefficient_semi_inf(xs_semi, ys_semi, epsilon):
    """Evaluate the shifted variable coefficient k(y)."""
    z = np.clip((ys_semi + 1.5) / epsilon, -60.0, 60.0)
    return 1 + 2 / (1 + np.exp(-z))


def generate_dataset_semi_inf(
    alpha=0.5,
    beta=5.0,
    epsilon=1.0,
    train_domain=(-5.0, 5.0, -5.0, 0.0),
    eval_domain=(-8.0, 8.0, -8.0, 0.0),
    n_obs_u=100,
    n_boundary_u=100,
    n_obs_k=100,
    n_pde=10_000,
    n_grid=300,
    sampling="uniform",
    sigma=2.5,
    exp_scale=7.0,
    device="cpu",
    dtype=torch.float32,
    plot=False,
    seed=1,
):
    """Generate observations, PDE points, and evaluation data."""
    rng = np.random.default_rng(seed)
    train_xmin, train_xmax, train_ymin, train_ymax = train_domain
    eval_xmin, eval_xmax, eval_ymin, eval_ymax = eval_domain

    def sample_points(n):
        if sampling == "uniform":
            x = rng.uniform(train_xmin, train_xmax, n)
            y = rng.uniform(train_ymin, train_ymax, n)
        elif sampling == "gaussian":
            x = rng.normal(0.0, sigma, n)
            y = -rng.exponential(exp_scale, n)
        elif sampling == "gaussian_exponential":
            x = rng.normal(0.0, sigma, n)
            y = -rng.exponential(exp_scale, n)
        else:
            raise ValueError(
                f"Unknown sampling '{sampling}'. Choose from: "
                "'uniform', 'gaussian', 'gaussian_exponential'."
            )
        return x, y

    xs, ys = sp.symbols("x y")
    alpha_s, beta_s = sp.symbols("alpha beta", positive=True)
    eps_s = sp.symbols("epsilon", positive=True)

    x_obs, y_obs = sample_points(n_obs_u)
    if sampling == "uniform":
        x_boundary = rng.uniform(train_xmin, train_xmax, n_boundary_u)
    else:
        x_boundary = rng.normal(0.0, sigma, n_boundary_u)
    y_boundary = np.full(n_boundary_u, train_ymax)
    x_obs_k, y_obs_k = sample_points(n_obs_k)
    x_pde, y_pde = sample_points(n_pde)

    u_obs = analytical_solution_semi_inf(x_obs, y_obs, alpha, beta)
    u_boundary = analytical_solution_semi_inf(x_boundary, y_boundary, alpha, beta)
    k_obs = coefficient_semi_inf(x_obs_k, y_obs_k, epsilon)
    f_function = source_term_semi_inf(xs, ys, alpha_s, beta_s, eps_s)
    f_values = f_function(x_pde, y_pde, alpha, beta, epsilon)

    x = np.linspace(eval_xmin, eval_xmax, n_grid)
    y = np.linspace(eval_ymin, eval_ymax, n_grid)
    X, Y = np.meshgrid(x, y)
    U = analytical_solution_semi_inf(X, Y, alpha, beta)
    K = coefficient_semi_inf(X, Y, epsilon)

    def tensor_points(x_values, y_values, requires_grad=True):
        return torch.tensor(
            np.column_stack((x_values, y_values)),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    X_obs = tensor_points(x_obs, y_obs)
    X_boundary = tensor_points(x_boundary, y_boundary)
    X_obs_k = tensor_points(x_obs_k, y_obs_k)
    X_pde = tensor_points(x_pde, y_pde)
    U_obs = torch.tensor(u_obs.reshape(-1, 1), dtype=dtype, device=device)
    U_boundary = torch.tensor(u_boundary.reshape(-1, 1), dtype=dtype, device=device)
    X_obs = torch.cat((X_obs, X_boundary), dim=0)
    U_obs = torch.cat((U_obs, U_boundary), dim=0)
    K_obs = torch.tensor(k_obs.reshape(-1, 1), dtype=dtype, device=device)
    F_pde = torch.tensor(f_values.reshape(-1, 1), dtype=dtype, device=device)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
        extent = [eval_xmin, eval_xmax, eval_ymin, eval_ymax]
        axes[0].imshow(U, extent=extent, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1].imshow(K, extent=extent, origin="lower", cmap="GnBu", vmin=1, vmax=3)
        axes[0].scatter(x_obs, y_obs, c=u_obs, cmap="RdBu_r", vmin=-1, vmax=1, s=20)
        axes[1].scatter(x_obs_k, y_obs_k, c=k_obs, cmap="GnBu", vmin=1, vmax=3, s=20)
        for axis in axes:
            axis.set_xlim(eval_xmin, eval_xmax)
            axis.set_ylim(eval_ymin, eval_ymax)
            axis.set_aspect("equal")
        plt.tight_layout()
        plt.show()

    return X_obs, U_obs, X_obs_k, K_obs, X_pde, F_pde, X, Y, U, K


def evaluate_model_semi_inf(
    model_u,
    model_k,
    analytical_solution=analytical_solution_semi_inf,
    coefficient=coefficient_semi_inf,
    X_obs=None,
    X_obs_k=None,
    train_domain=(-5.0, 5.0, -5.0, 0.0),
    eval_domain=(-8.0, 8.0, -8.0, 0.0),
    n_grid=400,
    alpha=0.5,
    beta=5.0,
    epsilon=1.0,
    device="cpu",
    plot=False,
    verbose=False,
    save_results=False,
    results_dir="results",
    save_plot=False,
    plot_path=None,
):
    """Evaluate semi-infinite-domain predictions and MAE inside/outside training."""
    train_xmin, train_xmax, train_ymin, train_ymax = train_domain
    eval_xmin, eval_xmax, eval_ymin, eval_ymax = eval_domain
    x = np.linspace(eval_xmin, eval_xmax, n_grid)
    y = np.linspace(eval_ymin, eval_ymax, n_grid)
    X, Y = np.meshgrid(x, y)
    U = analytical_solution(X, Y, alpha, beta)
    K_exact = coefficient(X, Y, epsilon)
    X_test = torch.tensor(
        np.column_stack((X.ravel(), Y.ravel())),
        dtype=torch.float32,
        device=device,
    )

    was_training_u, was_training_k = model_u.training, model_k.training
    model_u.eval()
    model_k.eval()
    with torch.no_grad():
        U_pred = model_u(X_test).detach().cpu().numpy().reshape(U.shape)
        K_pred = model_k(X_test).detach().cpu().numpy().reshape(K_exact.shape)
    if was_training_u:
        model_u.train()
    if was_training_k:
        model_k.train()

    error_u = np.abs(U_pred - U)
    error_k = np.abs(K_pred - K_exact)
    outside_mask = (
        (X < train_xmin) | (X > train_xmax)
        | (Y < train_ymin) | (Y > train_ymax)
    )
    inside_mask = ~outside_mask

    metrics = {
        "err_u_global": np.mean(error_u),
        "err_k_global": np.mean(error_k),
        "err_u_inside": np.mean(error_u[inside_mask]),
        "err_k_inside": np.mean(error_k[inside_mask]),
        "err_u_outside": np.mean(error_u[outside_mask]),
        "err_k_outside": np.mean(error_k[outside_mask]),
        "U_pred": U_pred,
        "K_pred": K_pred,
        "error_u": error_u,
        "error_k": error_k,
        "X": X,
        "Y": Y,
        "outside_mask": outside_mask,
    }

    if verbose:
        print("Semi-infinite-domain spatial generalization (MAE)")
        for name in ("u_global", "k_global", "u_inside", "k_inside", "u_outside", "k_outside"):
            prefix, region = name.split("_")
            print(f"{prefix} ({region}): {metrics['err_' + name]:.3e}")

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
        extent = [eval_xmin, eval_xmax, eval_ymin, eval_ymax]
        for axis, values, title, cmap in zip(
            axes.ravel(),
            (U_pred, error_u, K_pred, error_k),
            ("u prediction", "u absolute error", "k prediction", "k absolute error"),
            ("RdBu_r", "magma", "GnBu", "magma"),
        ):
            axis.imshow(values, extent=extent, origin="lower", cmap=cmap, aspect="equal")
            axis.set_title(title)
            axis.set_xlim(eval_xmin, eval_xmax)
            axis.set_ylim(eval_ymin, eval_ymax)
        plt.tight_layout()
        if save_plot:
            if plot_path is None:
                os.makedirs(results_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(results_dir, f"semi_infinite_prediction_{timestamp}.png")
            else:
                directory = os.path.dirname(plot_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()

    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(results_dir, f"semi_infinite_problem_results_{timestamp}.txt")
        with open(filepath, "w") as file:
            file.write("Semi-infinite-domain inverse problem (MAE)\n")
            for key in ("err_u_global", "err_k_global", "err_u_inside", "err_k_inside", "err_u_outside", "err_k_outside"):
                file.write(f"{key}: {metrics[key]:.8e}\n")

    return metrics


def compute_training_errors_semi_inf(
    model_u,
    model_k,
    X_obs,
    X_obs_k,
    analytical_solution=analytical_solution_semi_inf,
    coefficient=coefficient_semi_inf,
    alpha=0.5,
    beta=5.0,
    epsilon=1.0,
    device="cpu",
):
    """Compute relative L2 errors at semi-infinite training observations."""
    X_obs = torch.as_tensor(X_obs, dtype=torch.float32, device=device)
    X_obs_k = torch.as_tensor(X_obs_k, dtype=torch.float32, device=device)
    u_exact = torch.as_tensor(
        analytical_solution(
            X_obs[:, 0].detach().cpu().numpy(),
            X_obs[:, 1].detach().cpu().numpy(),
            alpha,
            beta,
        ),
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    k_exact = torch.as_tensor(
        coefficient(
            X_obs_k[:, 0].detach().cpu().numpy(),
            X_obs_k[:, 1].detach().cpu().numpy(),
            epsilon,
        ),
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    was_training_u, was_training_k = model_u.training, model_k.training
    model_u.eval()
    model_k.eval()
    with torch.no_grad():
        u_pred = model_u(X_obs)
        k_pred = model_k(X_obs_k)
    if was_training_u:
        model_u.train()
    if was_training_k:
        model_k.train()
    err_u = (torch.linalg.norm(u_pred - u_exact) / (torch.linalg.norm(u_exact) + 1e-14)).item()
    err_k = (torch.linalg.norm(k_pred - k_exact) / (torch.linalg.norm(k_exact) + 1e-14)).item()
    return err_u, err_k


# Short aliases for callers that import the module by problem type.
source_term = source_term_semi_inf
analytical_solution = analytical_solution_semi_inf
coefficient = coefficient_semi_inf
generate_dataset = generate_dataset_semi_inf
evaluate_model = evaluate_model_semi_inf
compute_training_errors = compute_training_errors_semi_inf