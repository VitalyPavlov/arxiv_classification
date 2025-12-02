from setuptools import setup, find_packages

setup(
    name="arxiv_classification",
    version="0.0.1",
    author="Vitaly Pavlov",
    description="Classification of arxiv articles",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy==2.0.2",
        "pandas==2.3.3",
        "torch==2.8.0",
        "ipykernel==6.30.1",
        "pytest==8.4.2",
        "black==25.9.0",
        "matplotlib==3.9.4",
        "omegaconf==2.3.0",
        "polars==1.35.2",
        "transformers==4.57.3",
        "accelerate==1.10.1",
        "pydantic==2.12.5",
        "jsonformer==0.12.0",
        "spacy==3.8.11",
        "nltk==3.9.2",
    ],
    python_requires=">=3.9",
    include_package_data=True,
)
