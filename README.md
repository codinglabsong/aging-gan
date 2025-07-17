different lr for gen and disc

# Aging GAN

Aging GAN is a research project exploring facial age transformation with a CycleGAN‑style approach. The model trains two ResNet‑style "encoder–residual–decoder" generators and two PatchGAN discriminators on the UTKFace dataset, split into **Young** and **Old** subsets. The generators learn to translate between these domains, effectively "aging" or "de-aging" a face image.

This repository contains training scripts, helper utilities, and inference scripts.

## Features

- **Unpaired Training Data** – splits the UTKFace dataset into *Young* (18‑28) and *Old* (40+) subsets and builds an unpaired `DataLoader`.
- **CycleGAN Architecture** – residual U‑Net generators and PatchGAN discriminators.
- **Training Utilities** – gradient clipping, separate generator/discriminator learning rates with linear decay, mixed precision via `accelerate`, and optional S3 checkpoint archiving.
- **Evaluation** – FID metric computation on the validation and test sets.
- **Weights & Biases Logging** – track losses and metrics during training.
- **Scriptable Workflows** – shell scripts for training and inference.
- **Sample Generation** – saves example outputs after each epoch.

## Installation

```bash
pip install -r requirements.txt
```

Optional development tools:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

The package itself can be installed with:

```bash
pip install -e .
```

## Data
Place the aligned UTKFace images under `data/utkface_aligned_cropped/UTKFace`.
The `prepare_dataset` function builds deterministic train/val/test splits and applies random flipping, cropping and rotation for training. 
Each split is divided into *Young* and *Old* subsets for unpaired training.

## Training
Run training with default hyper‑parameters:

```bash
bash scripts/run_train.sh --num_train_epochs 2 --train_batch_size 4
```

Additional options are available via:

```bash
python -m aging_gan.train --help
```

## Inference
Generate aged faces using the command-line helper:

```bash
bash scripts/run_inference.sh --input myface.jpg --direction young2old
```
The script loads `outputs/checkpoints/best.pth` by default and saves the result beside the input.

## AWS Utilities
When running on EC2, pass `--archive_and_terminate_ec2` to automatically sync `outputs/` to S3 and terminate the instance after training.

## Results
*Results will be added here once experiments are complete.*

## Example Outputs
*Example images will be shown here in a future update.*

## Repository Structure

- `src/aging_gan/` – core modules (`train.py`, `model.py`, etc.)
- `scripts/` – helper scripts for training and inference
- `notebooks/` – exploratory notebooks

## Requirements

- Python ≥ 3.10
- PyTorch
- Additional packages listed in `requirements.txt`

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).