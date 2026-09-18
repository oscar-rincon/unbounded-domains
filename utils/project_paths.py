from pathlib import Path


EXPERIMENT_ALIASES = {
    "manufactured_solution": "01_manufactured_solution",
    "hyperparameter_tuning": "02_hyperparameter_tunning",
    "hyperparameter_tunning": "02_hyperparameter_tunning",
    "individual_prediction": "03_individual_prediction",
    "simple_prediction": "03_individual_prediction",
    "sampling_approximation": "04_sampling_approximation",
    "kan_accuracy_efficiency": "05_kan_accuracy_efficiency",
}


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root by walking up until the project markers are found."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        if (path / "utils").is_dir() and (path / "main").is_dir():
            return path
    raise FileNotFoundError("Could not locate the repository root from the provided path.")


PROJECT_ROOT = find_repo_root(Path(__file__))
UTILS_DIR = PROJECT_ROOT / "utils"
EXPERIMENTS_DIR = PROJECT_ROOT / "main"


def normalize_experiment_name(name: str) -> str:
    """Resolve canonical experiment directory names from stable aliases."""
    if name in EXPERIMENT_ALIASES.values():
        return name
    if name in EXPERIMENT_ALIASES:
        return EXPERIMENT_ALIASES[name]
    raise KeyError(f"Unknown experiment name: {name}")


def utils_dir(repo_root: Path | None = None) -> Path:
    """Return the shared utilities directory."""
    return (repo_root or PROJECT_ROOT) / "utils"


def experiment_dir(name: str, repo_root: Path | None = None) -> Path:
    """Return the canonical directory for an experiment."""
    root = repo_root or PROJECT_ROOT
    return root / "main" / normalize_experiment_name(name)


def data_dir(name: str, repo_root: Path | None = None) -> Path:
    """Return the data directory for an experiment."""
    return experiment_dir(name, repo_root) / "data"


def figures_dir(name: str, repo_root: Path | None = None) -> Path:
    """Return the figures directory for an experiment."""
    return experiment_dir(name, repo_root) / "figures"


def results_dir(name: str, repo_root: Path | None = None) -> Path:
    """Return the results directory for an experiment."""
    return experiment_dir(name, repo_root) / "results"
