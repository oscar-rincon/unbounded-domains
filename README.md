# unbounded-domains

## Installation

We recommend setting up a new Python environment with conda. You can do this by running the following commands:

```
conda env create -f PIKAN-unbounded-domains.yml
conda activate PIKAN-unbounded-domains-env
```

### Install PyTorch with CUDA support

After activating the environment, install PyTorch, TorchVision, and TorchAudio with CUDA 12.8 support (adjust if your nvidia-smi shows a different CUDA version): 

 ```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
 ```

Make sure your system’s NVIDIA driver and CUDA toolkit are properly installed.
You can check your CUDA version with:

 ```
nvidia-smi
 ```

Example output: 

 ```
CUDA Version: 12.8
 ```

To confirm that PyTorch detects your GPU and CUDA correctly, run:

 ```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
 ```

## Repository structure

The repository is organized around shared utilities plus experiment-specific folders:

```
unbounded-domains/
├── utils/                      # Shared domain, PINN/KAN, plotting, and path helpers
├── main/
│   ├── 01_manufactured_solution/
│   ├── 02_hyperparameter_tunning/
│   ├── 03_individual_prediction/
│   ├── 04_sampling_approximation/
│   └── 05_kan_accuracy_efficiency/
├── figures/                    # Paper-level figures shared outside a single experiment
└── PIKAN-unbounded-domains.yml
```

### Recommended module and folder conventions

- Keep reusable code in `utils/` and experiment execution in `main/`.
- Keep each experiment's `data/`, `results/`, and `figures/` inside its own numbered folder.
- Use `utils/project_paths.py` for new scripts and notebooks instead of hardcoded relative paths.
- Treat these stable experiment aliases as the canonical names in code: `manufactured_solution`, `hyperparameter_tuning`, `individual_prediction`, `sampling_approximation`, and `kan_accuracy_efficiency`.

### Path helper usage

New scripts and notebooks should resolve paths through `utils/project_paths.py` so they keep working even when launched from a different working directory. This also hides legacy directory names such as `02_hyperparameter_tunning` behind stable aliases.

