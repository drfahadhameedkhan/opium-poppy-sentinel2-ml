from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="opium-poppy-sentinel2-ml",
    version="0.1.0",
    author="Fahad Hameed Khan",
    author_email="drfahadhameedkhan@gmail.com",
    description="Sentinel-2 ML for opium poppy monitoring in Afghanistan & Pakistan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drfahadhameedkhan/opium-poppy-sentinel2-ml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "tensorflow>=2.10.0",
        "earthengine-api>=0.1.370",
        "geopandas>=0.13.0",
    ],
)
