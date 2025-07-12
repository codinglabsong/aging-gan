added spectral norm to path gan.
learning reduces to 0.1
different lr for gen and disc


# Aging GAN

Aging GAN is a research project exploring face aging with a CycleGAN-style architecture. The code trains two U-Net generators and two PatchGAN discriminators on the CelebA dataset, preprocessing into **Young** and **Old** subsets. The generators learn to translate between these domains, effectively "aging" or "de-aging" a face image.

This repository contains training scripts, minimal utilities, and example notebooks.

## Features

- **Unpaired Training Data** – automatically split the CelebA dataset into Young vs. Old and create an unpaired `DataLoader`.
- **CycleGAN Architecture** – U-Net generators with ResNet encoders and PatchGAN discriminators.
- **Training Utilities** – gradient clipping, learning-rate scheduling, mixed-precision via `accelerate`, and optional encoder freezing.
- **Evaluation** – FID metric computation on the validation set.
- **Weights & Biases Logging** – track losses and metrics during training.
- **Scriptable Workflows** – run training from the command line with `scripts/run_train.sh`.

*Placeholders:* inference helpers, web demo, and quantitative results will be added later.

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
The `prepare_dataset` function downloads CelebA automatically and creates train, validation, and test splits. Images are center‑cropped and resized to 256×256. Each split is divided into *Young* and *Old* subsets for unpaired training.

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
The `aging_gan.inference` module is currently a stub. Once implemented, you will be able to generate aged faces from the command line using `scripts/run_inference.sh`.

## Results
*Results will be added here once experiments are complete.*

## Example Outputs
*Example images will be shown here in a future update.*

## Repository Structure

- `src/aging_gan/` – core modules (`train.py`, `model.py`, etc.)
- `scripts/` – helper shell scripts for training and (placeholder) inference
- `notebooks/` – exploratory notebooks

## Requirements

- Python ≥ 3.10
- PyTorch
- Additional packages listed in `requirements.txt`

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).