# Reproducible environment

This file records both the local validation environment and the final remote
training environment. The adapter and reported validation results were produced
in the ModelScope environment described below.

## Final ModelScope training environment

| Component | Version / value |
| --- | --- |
| Platform | ModelScope DSW, Linux |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| GPU | NVIDIA A10 |
| Available VRAM | approximately 22.18 GiB |
| BF16 support | Yes |
| ms-swift | 4.4.0 |
| Transformers | 5.8.1 |
| PEFT | 0.19.1 |
| bitsandbytes | 0.49.2 |
| Final precision | BF16 |
| Random seed / data seed | 42 / 42 |
| ModelScope repository | `google/gemma-3-270m-it` |
| Remote model path | `/mnt/workspace/logo-gemma-lora/models/gemma-3-270m-it` |

NF4 4-bit loading was tested but rejected for the final run because the
unmodified Q4 base model itself entered repetitive generation and failed to
close SVG output. The final selected experiment therefore uses the original
BF16 base weights. Runtime adapter loading also behaved inconsistently in this
package combination, so diagnostic evaluation was performed after merging the
adapter into a temporary copy of the BF16 base model. The merged model is not
part of the submission.

## Local validation software

| Component | Version |
| --- | --- |
| Operating system | Windows, 64-bit |
| Python | 3.10.19 (Anaconda build) |
| PyTorch | 2.5.1 |
| CUDA runtime reported by PyTorch | 12.1 |
| cuDNN | 9.1.0 |
| ms-swift | 4.4.0 |
| ModelScope | 1.38.1 |
| Transformers | 5.12.1 |
| PEFT | 0.19.1 |
| TRL | 0.29.1 |
| Accelerate | 1.14.0 |
| Datasets | 4.8.4 |
| Safetensors | 0.8.0 |

The exact core Python dependencies are pinned in `requirements.txt`.

## Local validation hardware

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce MX450 |
| VRAM | 2048 MiB |
| NVIDIA driver | 527.99 |
| CUDA compute capability | 7.5 |

The local GPU is suitable for environment and loading checks. Final LoRA
training may be run on a larger AI Studio GPU; its GPU and driver information
must then be added to the experiment report.

## Base model

| Item | Value |
| --- | --- |
| ModelScope repository | `google/gemma-3-270m-it` |
| Local project path | `models/gemma-3-270m-it` |
| Architecture | `Gemma3ForCausalLM` |
| Parameter count | 268,098,176 |
| Downloaded snapshot size | 575,486,395 bytes |
| `model.safetensors` SHA-256 | `700B710A9A99C295ED546647AA81CACF9F81F4C573EA2BE613A0E2517A44AFAB` |

The model directory is intentionally excluded from Git. Recreate it with:

```powershell
modelscope download --model google/gemma-3-270m-it `
  --local_dir models/gemma-3-270m-it
```

## Randomness policy

The project-wide random seed is **42**. Training and evaluation code should
apply it consistently to Python, NumPy, PyTorch, CUDA, the data loader, and
generation where the relevant API accepts a seed. Deterministic algorithms
should be enabled when practical; operations that cannot be deterministic must
be documented in `report.md`.

Expected seed setup:

```python
import random

import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

## Environment creation

The verified local environment reuses the CUDA-enabled packages from the
existing `dl` Conda environment:

```powershell
conda run -n dl python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a clean Linux or AI Studio environment, first install the PyTorch build
matching that machine's CUDA version, then install the other pinned packages.
Do not assume that the local CUDA 12.1 wheel is appropriate for every GPU
runtime.

## Verification

Run the following after recreating the environment:

```powershell
.\.venv\Scripts\python.exe -c "import torch, transformers, peft; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformers.__version__, peft.__version__)"
.\.venv\Scripts\python.exe -m pip check
```

At capture time, `torch.cuda.is_available()` returned `True`, all 219 training
rows and 17 validation rows had the expected system/user/assistant structure,
and the downloaded model completed a local forward pass successfully.
