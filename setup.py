from setuptools import setup, find_packages

setup(
    name="bart_reddit_lora",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # "python-dotenv",
        # "numpy",
        # "wandb",
        # "gradio",
        "ipykernel",
        # "huggingface_hub",
        "torch",
        "torchvision",
    ],
)