# Ethical Considerations

This project trains generative adversarial networks to translate face images between "young" and "old" domains. The code is research oriented and makes use of the publicly available **UTKFace** dataset. The following points summarize the main ethical issues and recommended practices.

## Data Source
* **UTKFace** images contain photos of real people. The dataset is distributed for academic purposes. Users must follow the dataset's license and terms of use.
* Data were center cropped and resized to 256×256 pixels. Only age labels were used to create two groups: ages 18‑28 and ages 40 and above.

## Intended Use
* The software is meant for research and educational exploration of image‑to‑image translation techniques.
* It should **not** be used for deception, surveillance, or impersonation. Generated images may not accurately reflect the depicted person.

## Bias and Limitations
* UTKFace is not perfectly balanced across ethnicities or genders. Models trained on it may reflect these imbalances.
* The approach does not guarantee realistic aging effects and may fail on extreme poses or occlusions.

## Responsible Usage
* Do not use the model to create misleading or defamatory content.
* Clearly disclose synthetic images in any public communication.
* Comply with privacy regulations and obtain consent before processing personal images.

## Contact
Questions or concerns can be directed to the project maintainers through the repository's issue tracker.
