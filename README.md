# RBDash v1.2



## install 
1. Clone this repository and navigate to RBDash folder
```bash
git clone https://github.com/rbdash.git
cd RBDash
```
2. Install Package
```bash
conda create -n rbdash python=3.10 -y
conda activate rbdash
pip install --upgrade pip
pip install -e .
```
3. 
- Install additional packages for training cases
```bash
pip install ninja
pip install flash-attn --no-build-isolation
``` 
- or install the specific version of flash_attn from the .whl file:
If you have already downloaded the flash_attn wheel file, for example, flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl, you can install it with the following command:
```bash
pip install flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Upgrade to latest code base
```bash
git pull
pip uninstall transformers
pip install -e .
```

## Pretrained Weights
We recommend users to download the pretrained weights from the following link [OpenCLIP-ConvNeXt-L](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup), [InternViT-6B-448px-V1-5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5/tree/main),and put them in model_zoo following Structure.
### Structure
```bash
RBDASH
├── rbdash
├── scripts
├── model_zoo
│   ├── OpenAI
│   │   ├── InternViT-6B-448px-V1-5
│   │   ├── openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup
│   │   ├── ...

```
## Model Zoo
[RBDash-v1.2-72b](https://huggingface.co/RBDash-Team/RBDash-v1.2-72b)

## Evaluation
In RBDash, we evaluate models on MME.
### MME
1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to ./rbdash-Eval/MME/MME_Benchmark_release_version.
3. Downloaded and put the weights to ./models/RBDash-v1.2-72b
4. inference and evaluate.
```bash
bash scripts/rbdash/eval/mme.sh
```
