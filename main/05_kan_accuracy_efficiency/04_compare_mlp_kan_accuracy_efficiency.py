"""Compare MLP and KAN accuracy and efficiency.

This script is the non-interactive counterpart of the companion notebook.
It saves tradeoff.svg/.pdf and far_field.svg/.pdf in the figures directory.
"""

import json
import os
import sys
from importlib import reload
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
UTILITIES_DIR = SCRIPT_DIR.parents[1] / "utils"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

import infinite
import pinns
import plotting

reload(infinite)
reload(pinns)
reload(plotting)

from infinite import analytical_solution_inf
from pinns import build_models, build_models_KAN, set_seed
from project_paths import experiment_dir, figures_dir, find_repo_root
from figure_far_field import plot_far_field
from figure_tradeoff import plot_tradeoff


REPO_ROOT = find_repo_root(SCRIPT_DIR)
EXPERIMENT_DIR = experiment_dir("kan_accuracy_efficiency", REPO_ROOT)
MLP_RESULTS_DIR = EXPERIMENT_DIR / "results_accuracy_efficiency_2026-09-14_11-02-44"
KAN_RESULTS_DIR = EXPERIMENT_DIR / "results_accuracy_efficiency_2026-09-14_11-13-10"
TARGET_ERROR = 1e-2
FIGURES_DIR = figures_dir("kan_accuracy_efficiency", REPO_ROOT)


def load_experiment_data():
    """Load and combine the MLP and KAN summary metrics."""
    mlp = pd.read_csv(MLP_RESULTS_DIR / "summary_metrics.csv")
    kan = pd.read_csv(KAN_RESULTS_DIR / "summary_metrics.csv")
    mlp.columns = mlp.columns.str.strip()
    kan.columns = kan.columns.str.strip()
    for frame in (mlp, kan):
        frame["err_global_mean"] = frame[["err_u_global", "err_k_global"]].mean(axis=1)
    combined = pd.concat([mlp, kan], ignore_index=True)
    print(f"MLP runs: {len(mlp)}")
    print(f"KAN runs: {len(kan)}")
    print(f"Total runs: {len(combined)}")
    return combined


def select_model_runs(data):
    """Select the smallest-parameter run meeting the target error."""
    selected = {}
    for model_name in ("MLP", "KAN"):
        runs = data[data["model_type"] == model_name].sort_values("parameters")
        achieved = runs[runs["err_global_mean"] <= TARGET_ERROR]
        if achieved.empty:
            row = runs.loc[(runs["err_global_mean"] - TARGET_ERROR).abs().idxmin()]
        else:
            row = achieved.iloc[0]
        selected[model_name] = row
        print(f"\nSelected {model_name}:")
        print(row)
    return selected


def load_models(selected, device):
    """Build selected architectures and load their trained weights."""
    model_paths = {}
    for model_name, run_root, builder in (
        ("MLP", MLP_RESULTS_DIR, build_models),
        ("KAN", KAN_RESULTS_DIR, build_models_KAN),
    ):
        run_dir = run_root / model_name / str(selected[model_name]["timestamp"])
        with open(run_dir / "run_metrics_and_config.json", "r") as file:
            config = json.load(file)

        if model_name == "MLP":
            model_u, model_k = builder(
                device=device,
                hidden_layers=config["hidden_layers"],
                hidden_units=config["hidden_units"],
                activation=nn.Tanh(),
            )
        else:
            model_u, model_k = builder(
                device=device,
                hidden_layers=config["hidden_layers"],
                hidden_units=config["hidden_units"],
                grid_size=config["grid_size"],
                spline_order=config["spline_order"],
            )

        model_u.load_state_dict(torch.load(run_dir / "model_u.pt", map_location=device))
        model_k.load_state_dict(torch.load(run_dir / "model_k.pt", map_location=device))
        model_u.eval()
        model_k.eval()
        model_paths[model_name] = (model_u, model_k, config)
    return model_paths


def evaluate_models(models, device):
    """Evaluate both selected u-models on the common far-field grid."""
    eval_min, eval_max, n_grid = -50.0, 50.0, 400
    x = np.linspace(eval_min, eval_max, n_grid)
    y = np.linspace(eval_min, eval_max, n_grid)
    X, Y = np.meshgrid(x, y)
    config = models["MLP"][2]
    U = analytical_solution_inf(X, Y, config["pde_alpha"], config["pde_beta"])
    X_test = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

    with torch.no_grad():
        U_pred_mlp = models["MLP"][0](X_test).cpu().numpy().reshape(U.shape)
        U_pred_kan = models["KAN"][0](X_test).cpu().numpy().reshape(U.shape)

    outside_mask = (np.abs(X) > 5.0) | (np.abs(Y) > 5.0)
    error_mlp = np.abs(U_pred_mlp - U)
    error_kan = np.abs(U_pred_kan - U)
    print("\n========================================")
    print("Far-field evaluation")
    print("========================================")
    print(f"MLP outside MAE: {np.mean(error_mlp[outside_mask]):.4e}")
    print(f"KAN outside MAE: {np.mean(error_kan[outside_mask]):.4e}")
    return U_pred_mlp, U_pred_kan, U, x, y


def main():
    set_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_experiment_data()
    plot_tradeoff(
        data[data["model_type"] == "MLP"],
        data[data["model_type"] == "KAN"],
        output_dir=FIGURES_DIR,
        target_error=TARGET_ERROR,
    )
    selected = select_model_runs(data)
    models = load_models(selected, device)
    U_pred_mlp, U_pred_kan, U, x, y = evaluate_models(models, device)
    plot_far_field(
        U_pred_mlp,
        U_pred_kan,
        U,
        x,
        y,
        palette={"kan": "#1F77B4", "mlp": "#333333"},
        output_dir=FIGURES_DIR,
    )


if __name__ == "__main__":
    main()
