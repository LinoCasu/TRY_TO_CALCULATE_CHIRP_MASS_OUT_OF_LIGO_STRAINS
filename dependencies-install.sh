#!/bin/bash
echo "Creating a Conda environment 'ligo_env' with Python 3.10..."
conda create -n ligo_env python=3.10 -y
echo "Activating the environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"  # Falls erforderlich
conda activate ligo_env
echo "Installing PyCBC and lalsuite from conda-forge..."
conda install -c conda-forge pycbc lalsuite -y
echo "Dependencies have been installed in the 'ligo_env' Conda environment."
echo "To activate the environment in the future, run: conda activate ligo_env"
