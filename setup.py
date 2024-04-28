from setuptools import setup, find_packages

setup(
    name='mydataloader',
    version='0.1',
    packages=find_packages(include=['mydataloader', 'mydataloader.*']),
    install_requires=[
        'torch',
        'pandas',
        'pyarrow'
    ],
    description='Package to load data using PyTorch Dataloaders and Parquet files'
)
