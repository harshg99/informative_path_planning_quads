#!/bin/bash --login
source ~/.bashrc

#installing conda
conda create -n torch python=3.8
source /miniconda/etc/profile.d/conda.sh
conda activate torch

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

mkdir /home/code
cd /home/code

git clone https://github.com/ljarin/Active-sampling-with-motion-primitives.git
cd Active-sampling-with-motion-primitives
git checkout Search_vGYM
pip install -r requirements.txt

cd /home
git clone https://github.com/ljarin/motion_primitives.git
cd motion_primitives/motion_primitives_py && pip3 install -e .
