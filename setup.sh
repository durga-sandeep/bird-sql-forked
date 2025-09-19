#!/bin/bash

# Exit on any error
set -e

echo "Starting system setup..."

echo "Step 1: Downloading and installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

echo "Step 2: Setting up PATH and sourcing bashrc..."
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

echo "Step 3: Installing huggingface_hub..."
pip install huggingface_hub

echo "Step 4: Configuring git..."
cd bird-sql-forked
git config user.name "durga-sandeep"
git config user.email "durga@distyl.ai"

echo "Step 5: Creating conda environment..."
conda create -n bird-sql python=3.10

echo "Step 6: Initializing conda for bash..."
conda init bash

echo "Step 7: Installing NVIDIA drivers..."
sudo apt update && sudo apt install -y nvidia-driver-535 nvidia-utils-535

echo "Step 8: Loading NVIDIA kernel modules..."
sudo modprobe nvidia
sudo modprobe nvidia-drm

echo "Setup completed successfully!"
echo "Please restart your terminal or run 'source ~/.bashrc' to use conda commands."
