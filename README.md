# Aging GAN
Aging GAN is an unpaired image-to-image translation project for facial age transformation built on a CycleGAN-style framework. It provides end-to-end tooling—from data preprocessing to training, inference, and a live Gradio demo—while offering infrastructure utilities to simplify deployment on AWS. The model trains two ResNet‑style "encoder-residual-decoder" generators and two PatchGAN discriminators on the UTKFace dataset, split into **Young** and **Old** subsets. The generators learn to translate between these domains, effectively "aging" or "de-aging" a face image.

This repository contains training scripts, helper utilities, inference scripts, and a Gradio app for demo purposes.

## Features
- **CycleGAN Architecture** - ResNet‑style "encoder-residual-decoder" generators and PatchGAN discriminators.
- **Data Pipeline & Preprocessing** - Deterministic train/val/test splits, on-the-fly augmentations, unpaired DataLoader that pairs Young (18–28) and Old (40+) faces at each batch
- **Training Utilities, Efficiency, Stability** - gradient clipping to stabilize adversarial updates, separate generator/discriminator learning rates with linear decay for latter half of training, mixed precision via `accelerate` for 2× speed/memory improvements, and checkpoint archiving.
- **Evaluation** - FID (Frechet Inception Distance) evaluation on validation and test splits.
- **Weights & Biases Logging** - track losses and metrics during training.
- **Inference Utilities** - command-line interface for image generation on user input.
- **AWS Utilities** - When running on EC2, IAM-ready scripts to automatically sync `outputs/` to S3 when saving checkpoints and terminate the instance after training.
- **Gradio Demo** – ready-to-deploy web app for interactive predictions. Hosted on [Huggingface Spaces](https://huggingface.co/spaces/codinglabsong/Reddit-User-Mimic-Bot).
- **Scriptable Workflows** - shell scripts for training and inference.
- **Reproducible Workflows** - configuration-based training scripts and environment variables. Optionally set seed during training.
- **Developer Tools** - linting with ruff and black, plus  unit tests.

## Installation
1. Clone this repository and install the core dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. (Optional) Install development tools for linting and testing:
    ```bash
    pip install -r requirements-dev.txt
    pre-commit install
    ```

3. Install the package itself (runs `setup.py`):

    ```bash
    # Standard install:
    pip install .

    # Or editable/development install:
    pip install -e .
    ```

## Data
We leverage the UTKFace dataset—a public collection of over 20,000 face images with age annotations (0–116), then use the aligned and cropped version for consistency.

Place your data under:
> data/utkface_aligned_cropped/UTKFace

The prepare_dataset function handles:
- Age-based splits: thresholds at 18–28 for Young, 40+ for Old, ignoring mid-range ages.
- Deterministic shuffling: 80% train, 10% validation, 10% test with a fixed RNG seed.
- Augmentations (train only): random horizontal flips, resizing to img_size+50, center/random cropping to img_size, random rotations up to 80°, and normalization to [-1, 1].
- Evaluation transforms: resize → center crop → normalization (no randomness).

This pipeline ensures both diversity (via augmentation) and reproducibility (fixed splits and RNG).

## Usage
### Training
Use the wrapper script to launch training with your preferred hyper‑parameters:

```bash
bash scripts/run_train.sh --num_train_epochs 50 --lambda_cyc_value 4.0 --train_batch_size 4 --gen_lr 0.0002 --disc_lr 0.0002 --num_sample_generations_to_save 10 --archive_and_terminate_ec2
```

Additional options are available via:

```bash
python -m aging_gan.train --help
```

### Inference (CLI)
Generate aged faces using the command-line helper:

```bash
bash scripts/run_inference.sh --input path/to/face.jpg --direction young2old --ckpt outputs/checkpoints/best.pth
```
The script loads `outputs/checkpoints/best.pth` by default and saves the result beside the input.

## Results
*Results will be added here once experiments are complete.*

### Example Outputs
*Example images will be shown here in a future update.*

### Considerations for Improvements
This project intentionally focused more on the methods and pipeline than the actual results. In the case you have more time and resources to improve the model, good points to start are...

- Increase the dataset size by scraping or adding high quality Q/A data.
- Upgrade the model architecture to Bart-Large.
- Increase the LoRA rank (r).
- Train for more epochs.
- Use a better evaluation metric like RougeL for early stopping.

## Running the Gradio Inference App
This project includes an interactive Gradio app for making predictions with the trained model.

1. **Obtain the Trained Model:**
    - Ensure that a trained model directory (e.g., `outputs/bart-base-reddit-lora/`) is available in the project root.
    - If you trained the model yourself, it should be saved automatically in the project root.
    - Otherwise, you can download it from [Releases](https://github.com/codinglabsong/bart-reddit-lora/releases/tag/v1.0.0) and add it in the project root.

2. **Run the App Locally:**
    ```bash
    python app.py
    ```
    - Visit the printed URL (e.g., `http://127.0.0.1:7860`) to interact with the model.

> You can also access the hosted demo on [Huggingface Spaces](https://huggingface.co/spaces/codinglabsong/Reddit-User-Mimic-Bot)

## Testing
Run unit tests with:

```bash
pytest
```

## Repository Structure
- `src/aging_gan/` - core modules (`train.py`, `model.py`, etc.)
- `scripts/` - helper scripts for training and inference
- `notebooks/` - exploratory notebooks
- `tests/` - simple unit tests
- `outputs/` - generated directory for model outputs including `checkpoints/`, `images/` (for generated images per epoch), `metrics/test_metrics.json` (for storing test metrics for `best.pth` during training)

## Requirements
- Python ≥ 3.10
- PyTorch >= 2.6
- Additional packages listed in `requirements.txt`

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgements
- [Original GAN Paper](https://arxiv.org/abs/1406.2661)
- [Cycle-Consistent Adversarial Networks Paper](https://arxiv.org/abs/1703.10593)

## License
This project is licensed under the [MIT License](LICENSE).