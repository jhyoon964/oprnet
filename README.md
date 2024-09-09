# Multimodal 3D object detection

## Installation
The code is developed with **CUDA 11.1**, **Python >= 3.8.19**, **PyTorch >= 1.10.1**
```
conda create -n oprnet python=3.8
conda activate oprnet
conda install -y pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y pytorch3d -c pytorch3d
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 waymo-open-dataset-tf-2-2-0 nuscenes-devkit==1.0.5 spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
git clone https://github.com/Nightmare-n/GD-MAE
cd oprnet && python setup.py develop --user
cd pcdet/ops/dcn && python setup.py develop --user
```


## Training & Testing
```
# Train
bash scripts/dist_train.sh

# Test
bash scripts/dist_test.sh
