#!/bin/bash

# Clone this repo
git clone https://github.com/cuongvng/srgan-pytorch ~/srgan-pytorch && cd ~/srgan-pytorch/

# Download data: you can copy and paste all those lines at once
mkdir DIV2K && cd DIV2K
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2K_train_LR_bicubic_X4.zip && unzip DIV2K_train_HR.zip
unzip DIV2K_valid_LR_bicubic_X4.zip && unzip DIV2K_valid_HR.zip

# rm DIV2K_train_LR_bicubic_X4.zip DIV2K_train_HR.zip DIV2K_valid_HR.zip DIV2K_valid_LR_bicubic_X4.zip

# Install Conda
cd /tmp
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Then follow the instructions, type `ENTER` -> `yes` -> `ENTER` -> `yes`.

# Activate the environment and install required packages
source ~/.bashrc
conda create --name srgan-pytorch -y && conda activate srgan-pytorch
cd ~/srgan-pytorch/
conda install -c conda-forge python=3.7.9 -y
pip install --no-cache-dir -r requirements.txt
