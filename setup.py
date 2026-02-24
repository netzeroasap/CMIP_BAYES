from setuptools import setup, find_packages

setup(
    name="cmip_bayes",                  # Package name
    version="0.1",
    packages=find_packages(),           # Detects utils/ and modeling/
    install_requires=[
        "pymc",
        "numpy",
        "xarray",
        "matplotlib",
        "seaborn",
        "arviz",
        "pytensor",
        "pandas"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    description="Bayesian analysis of CMIP data",
    author="Kate Marvel",
    url="https://github.com/netzeroasap/CMIP_BAYES/"
)
