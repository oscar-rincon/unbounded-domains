"""Generate hyperparameter configurations and match KAN/MLP architectures."""

import json
import os
import sys
from importlib import reload
from itertools import product
from pathlib import Path

import torch
from calflops import calculate_flops


SCRIPT_DIR = Path(__file__).resolve().parent
UTILITIES_DIR = SCRIPT_DIR.parents[1] / "utils"
os.chdir(SCRIPT_DIR)
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

import infinite
import pinns
import plotting

reload(infinite)
reload(pinns)
reload(plotting)

from pinns import build_models, build_models_KAN, set_seed
from project_paths import data_dir, find_repo_root


KAN_SEARCH_SPACE = {
    "hidden_layers": [1, 2, 3],
    "hidden_units": [15, 25, 35],
    "grid_size": [3, 5, 7],
    "spline_order": [2, 3, 4],
    "learning_rate": [1e-4, 1e-3, 1e-2],
}

MLP_SEARCH_SPACE = {
    "hidden_layers": [1, 2, 3],
    "hidden_units": [25, 50, 75],
    "activation": ["Tanh", "SiLU", "Sigmoid"],
    "learning_rate": [1e-4, 1e-3, 1e-2],
}

KAN_DEPTHS = [1, 2, 3]
KAN_WIDTHS = [15, 25, 35]
KAN_GRIDS = [5]
SPLINE_ORDER = 3
MLP_SWEEP_WIDTHS = range(10, 150)
INPUT_SHAPE = (1, 2)
REPO_ROOT = find_repo_root(SCRIPT_DIR)
DATA_DIR = data_dir("hyperparameter_tuning", REPO_ROOT)


def generate_configurations():
    """Return the complete KAN and MLP Cartesian-product search spaces."""
    kan_configurations = [
        dict(zip(KAN_SEARCH_SPACE, values))
        for values in product(*KAN_SEARCH_SPACE.values())
    ]
    mlp_configurations = [
        dict(zip(MLP_SEARCH_SPACE, values))
        for values in product(*MLP_SEARCH_SPACE.values())
    ]
    return kan_configurations, mlp_configurations


def save_configurations(kan_configurations, mlp_configurations):
    """Save both configuration lists in the notebook data directory."""
    DATA_DIR.mkdir(exist_ok=True)
    paths = {
        "KAN": DATA_DIR / "kan_configurations.json",
        "MLP": DATA_DIR / "mlp_configurations.json",
    }
    for path, configurations in (
        (paths["KAN"], kan_configurations),
        (paths["MLP"], mlp_configurations),
    ):
        with path.open("w", encoding="utf-8") as file:
            json.dump(configurations, file, indent=2)
        print(f"Saved {len(configurations)} configurations to {path}")


def calculate_parameter_matches(device):
    """Match each selected KAN architecture to the closest MLP width."""
    kan_labels = []
    kan_params_list = []
    mlp_matched_widths = []
    mlp_matched_params_list = []

    for depth in KAN_DEPTHS:
        for units in KAN_WIDTHS:
            for grid in KAN_GRIDS:
                model_kan, _ = build_models_KAN(
                    device,
                    hidden_layers=depth,
                    hidden_units=units,
                    grid_size=grid,
                    spline_order=SPLINE_ORDER,
                )
                _, _, kan_params = calculate_flops(
                    model_kan,
                    input_shape=INPUT_SHAPE,
                    print_results=False,
                    print_detailed=False,
                    output_as_string=False,
                )

                best_width = None
                best_mlp_params = None
                min_difference = float("inf")
                for mlp_width in MLP_SWEEP_WIDTHS:
                    model_mlp, _ = build_models(
                        device,
                        hidden_layers=depth,
                        hidden_units=mlp_width,
                    )
                    _, _, mlp_params = calculate_flops(
                        model_mlp,
                        input_shape=INPUT_SHAPE,
                        print_results=False,
                        print_detailed=False,
                        output_as_string=False,
                    )
                    difference = abs(mlp_params - kan_params)
                    if difference < min_difference:
                        min_difference = difference
                        best_width = mlp_width
                        best_mlp_params = mlp_params

                kan_labels.append(f"L={depth}\nN={units}")
                kan_params_list.append(kan_params)
                mlp_matched_widths.append(best_width)
                mlp_matched_params_list.append(best_mlp_params)

    return kan_labels, kan_params_list, mlp_matched_widths, mlp_matched_params_list


def print_parameter_matches(matches):
    """Print the parameter-matching summary table."""
    kan_labels, kan_params, mlp_widths, mlp_params = matches
    print("\n" + "=" * 75)
    print("PARAMETER-MATCHED ARCHITECTURES")
    print("=" * 75)
    print(
        f"{'KAN':<12} | {'KAN Params':>12} | {'MLP Width':>10} | "
        f"{'MLP Params':>12} | {'Difference':>12}"
    )
    print("-" * 75)
    for label, kan_value, mlp_width, mlp_value in zip(
        kan_labels, kan_params, mlp_widths, mlp_params
    ):
        difference = abs(kan_value - mlp_value)
        print(
            f"{label.replace(chr(10), ', '):<12} | "
            f"{kan_value:>12,} | {mlp_width:>10} | "
            f"{mlp_value:>12,} | {difference:>12,}"
        )


def main():
    """Generate configurations, save them, and print architecture matches."""
    set_seed(1)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kan_configurations, mlp_configurations = generate_configurations()
    print(f"KAN candidate configurations: {len(kan_configurations)}")
    print(f"MLP candidate configurations: {len(mlp_configurations)}")
    save_configurations(kan_configurations, mlp_configurations)

    matches = calculate_parameter_matches(device)
    print_parameter_matches(matches)


if __name__ == "__main__":
    main()
