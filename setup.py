from setuptools import setup, find_packages

setup(
    name="aging_gan",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "python-dotenv",
        "numpy",
        "wandb",
        # "gradio",
        "ipykernel",
        "matplotlib",
        "torch",
        "torchvision",
        "accelerate",
        "segmentation-models-pytorch",
        "gdown",
        "tqdm",
        "torchmetrics",
    ],
)