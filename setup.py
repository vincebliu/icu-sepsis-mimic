from setuptools import setup, find_packages
import os

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='icu-sepsis-mimic',
    version='0.1.0',
    description='A tabular MDP benchmark for sepsis treatment optimization built from MIMIC-IV',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={
        'icu_sepsis_mimic': ['../data/*.parquet', '../data/*.pkl',
                              '../data/datasets/*.parquet'],
    },
    install_requires=[
        'gymnasium>=0.26.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'pyarrow>=6.0.0',
        'scikit-learn>=1.0.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
)
