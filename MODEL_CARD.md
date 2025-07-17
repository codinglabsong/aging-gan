# Model Card: Aging GAN

Aging GAN is a CycleGAN‑style model that translates face images between two age domains: **Young** (ages 18‑28) and **Old** (ages 40+). It consists of two U‑Net generators and two PatchGAN discriminators trained with adversarial, cycle‑consistency and identity losses.

## Intended Use
* Research and educational experiments on image‑to‑image translation and face aging.
* Not intended for production applications, facial recognition, or any form of impersonation or deceptive media generation.

## Dataset
* Training data is derived from the **UTKFace** dataset. Images were cropped and resized to 256×256 pixels.
* The dataset was split 80/10/10 into train/validation/test subsets with equal numbers of young and old images.

## Training
* Default training runs for 100 epochs using the Adam optimizer and mixed‑precision via Hugging Face Accelerate.
* Frechet Inception Distance (FID) is computed on the validation set each epoch; the model with the best FID score is saved.

## Limitations
* The model is limited to the diversity and quality of UTKFace and may not generalize to unseen demographics or lighting conditions.
* Generated aging effects are approximate and should not be considered authentic depictions of an individual.

## Ethical Considerations
See [ETHICS.md](ETHICS.md) for discussion of data usage, potential biases and recommended precautions.

## Citation
If you use this code or model in your research, please cite the repository and the UTKFace dataset.