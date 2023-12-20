# RBDash

## Install

1. Clone this repository and navigate to RBDash folder
```bash
git clone https://github.com/rbdash.git
cd RBDash
```

2. Install Package
```Shell
conda create -n rbdash python=3.10 -y
conda activate rbdash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install ninja
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip uninstall transformers
pip install -e .
```

## Model Zoo
[RBDash-v1-13b](https://huggingface.co/RBDash-Team/rbdash-v1-13b/tree/main)
## Evaluation
In RBDash, we evaluate models on MME.
### MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `./playground/data/eval/MME/MME_Benchmark_release_version`.
3. Downloaded and put the [weights](https://huggingface.co/RBDash-Team/rbdash-v1-13b/tree/main) to `./models/rbdash-v1-13b`
4. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/mme.sh
```
