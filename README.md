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