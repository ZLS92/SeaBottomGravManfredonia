#!/bin/bash

# This script sets up the environment on LINUX
# The enviroment was tested with the anaconda distribution v23.3.1: 
# https://www.anaconda.com/download

# =============================================================================
# CREATE THE CONDA ENVIRONMENT
# =============================================================================

# Default environment name
default_env_name="ManfredoniaGrav"

# Prompt the user for the environment name
read -p "Enter the name of the environment to create (press Enter to use '$default_env_name'): " env_name

# Use the default name if the user presses Enter
env_name=${env_name:-$default_env_name}

# Function to check if an environment exists
check_env_exists() {
    conda env list | grep -q "^$1 "
}

# Check if the environment already exists
if check_env_exists "$env_name"; then
    echo "Environment '$env_name' already exists. Exiting script."
    exit 1
fi

# Ensure the conda shell integration is loaded
CONDA_BASE=$(conda info --base)  # Get the base path of the conda installation
source "$CONDA_BASE/etc/profile.d/conda.sh"  # Source the conda.sh script

# -----------------------------------------------------------------------------
# Create and activate environment with Python version (i.e., 3.10)
# -----------------------------------------------------------------------------

echo "> Creating conda env with python=3.10 from conda-forge"
conda create -n "$env_name" python=3.10 -c conda-forge -y

echo "> Activating conda env"
conda activate "$env_name"

# Set channel priority to strict
echo "> Seting channel priority to strict"
conda config --env --set channel_priority strict

# -----------------------------------------------------------------------------
# Install core libraries from conda-forge (safe & compatible)
# -----------------------------------------------------------------------------

echo "> Installing scientific core packages"
conda install numpy scipy matplotlib shapely gdal pyproj -c conda-forge -y --repodata-fn=repodata.json
pip install harmonica

# -----------------------------------------------------------------------------
# Optional: GUI / plotting
# -----------------------------------------------------------------------------

echo "> Installing interactive plotting tools"
conda install ipympl ipywidgets -c conda-forge -y

# -----------------------------------------------------------------------------
# Logging / PDF creation
# -----------------------------------------------------------------------------

echo "> Installing logging / PDF tools"
pip install pdfkit
conda install -c conda-forge wkhtmltopdf -y

