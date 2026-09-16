"""PINN utilities for the semi-infinite manufactured problem.

The neural-network architectures, PDE residual, loss scheduling, and training
loop are shared with :mod:`pinns`. This module supplies the semi-infinite
problem data and keeps the familiar public names available with ``_semi_inf``
suffixes where the problem-specific behavior matters.
"""

import os
import time
from datetime import datetime

import torch.nn as nn

import pinns as _pinns
from semi_infinite import (
    analytical_solution_semi_inf,
    coefficient_semi_inf,
    compute_training_errors_semi_inf,
    evaluate_model_semi_inf,
    generate_dataset_semi_inf,
)

# Shared architectures, losses, schedulers, and saving helpers.
from pinns import *  # noqa: F401,F403


pde_loss_semi_inf = _pinns.pde_loss_inf


def compute_analytical_errors_semi_inf(
    model_u,
    model_k,
    pde_alpha=0.5,
    pde_beta=5.0,
    epsilon=1.0,
    device="cpu",
    **kwargs,
):
    """Compute semi-infinite analytical errors on the training observations."""
    return compute_training_errors_semi_inf(
        model_u=model_u,
        model_k=model_k,
        analytical_solution=analytical_solution_semi_inf,
        coefficient=coefficient_semi_inf,
        alpha=pde_alpha,
        beta=pde_beta,
        epsilon=epsilon,
        device=device,
        **kwargs,
    )


def train_dual_network_semi_inf(model_u, model_k, **kwargs):
    """Train the shared dual PINN loop with semi-infinite problem data."""
    kwargs.setdefault("n_boundary_u", kwargs.get("n_obs_u", 100))
    kwargs.setdefault("analytical_solution_inf", analytical_solution_semi_inf)
    kwargs.setdefault("coefficient_inf", coefficient_semi_inf)

    original_generator = _pinns.generate_dataset_inf
    _pinns.generate_dataset_inf = generate_dataset_semi_inf
    try:
        return _pinns.train_dual_network(model_u, model_k, **kwargs)
    finally:
        _pinns.generate_dataset_inf = original_generator


def run_experiment_semi_inf(
    model_type="MLP",
    hidden_layers=3,
    hidden_units=25,
    activation=nn.Tanh(),
    grid_size=5,
    spline_order=3,
    adam_lr=1e-3,
    adam_iters=2000,
    lbfgs_iters=2000,
    sigma=5.5,
    exp_scale=7.0,
    n_obs_u=100,
    n_obs_k=100,
    n_pde=1000,
    seed=2,
    alpha=0.5,
    beta=5.0,
    epsilon=1.0,
    results_dir="results",
    device="cpu",
):
    """Build, train, evaluate, and summarize a semi-infinite PINN run."""
    start_time = time.time()
    model_upper = model_type.upper()

    if model_upper == "KAN":
        model_u, model_k = build_models_KAN(
            device=device,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            grid_size=grid_size,
            spline_order=spline_order,
        )
    else:
        model_u, model_k = build_models(
            device=device,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            activation=activation,
        )

    total_params = sum(p.numel() for p in model_u.parameters())
    total_params += sum(p.numel() for p in model_k.parameters())
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_path = os.path.join(results_dir, model_upper)

    history = train_dual_network_semi_inf(
        model_u=model_u,
        model_k=model_k,
        adam_lr=adam_lr,
        adam_iters=adam_iters,
        lbfgs_iters=lbfgs_iters,
        sampling="gaussian_exponential",
        sigma=sigma,
        exp_scale=exp_scale,
        n_obs_u=n_obs_u,
        n_boundary_u=n_obs_u,
        n_obs_k=n_obs_k,
        n_pde=n_pde,
        seed=seed,
        save_results=True,
        base_dir=base_output_path,
        run_name=timestamp,
        analytical_solution_inf=analytical_solution_semi_inf,
        coefficient_inf=coefficient_semi_inf,
        pde_alpha=alpha,
        pde_beta=beta,
        epsilon=epsilon,
        device=device,
    )

    eval_results = evaluate_model_semi_inf(
        model_u=model_u,
        model_k=model_k,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        device=device,
    )

    timing_results = benchmark_model_derivatives(model_u=model_u, device=device)
    err_u = eval_results["err_u_global"]
    err_k = eval_results["err_k_global"]
    run_dir = os.path.join(base_output_path, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    history["run_dir"] = run_dir

    return {
        "model_u": model_u,
        "model_k": model_k,
        "history": history,
        "evaluation": eval_results,
        "timing": timing_results,
        "parameters": total_params,
        "err_u_global": err_u,
        "err_k_global": err_k,
        "mean_global_error": 0.5 * (err_u + err_k),
        "training_time_sec": history.get("training_time_sec"),
        "compute_time_sec": time.time() - start_time,
        "timestamp": timestamp,
        "run_dir": run_dir,
    }


# Explicit aliases make the module convenient in notebooks that use generic names.
train_dual_network = train_dual_network_semi_inf
run_experiment = run_experiment_semi_inf
analytical_solution = analytical_solution_semi_inf
coefficient = coefficient_semi_inf
generate_dataset = generate_dataset_semi_inf
evaluate_model = evaluate_model_semi_inf
compute_training_errors = compute_training_errors_semi_inf
