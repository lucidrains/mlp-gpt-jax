from setuptools import setup, find_packages

setup(
    name="mlp-gpt-jax",
    packages=find_packages(),
    version="0.0.19",
    license="MIT",
    description="MLP GPT - Jax",
    author="Phil Wang",
    author_email="",
    url="https://github.com/lucidrains/mlp-gpt-jax",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "language model",
        "multilayered-perceptron",
        "jax"
    ],
    install_requires=[
        "click",
        "click-option-group",
        "einops>=0.3",
        "dm-haiku",
        "jax",
        "jaxlib",
        "optax",
        "torch",
        "tqdm"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
