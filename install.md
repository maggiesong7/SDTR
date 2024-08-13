
## Environments  
  Linux, Python==3.7.6, CUDA == 11.3, pytorch == 1.9.0, mmdet3d == 0.17.1   

## Creat virtual environment
```bash
conda create -n sdtr python=3.7.6 -y
conda activate sdtr
```

## Install Torch
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.9.0 torchvision==0.10.0
conda install cudatoolkit==11.3.1 cudnn
```

## Install MMCV
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0/index.html
```
## Install MMDet and MMSegmentation.
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mmdet mmsegmentation
```

## Install MMDetection3D
```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -v -e .
cd ..
```

## Install SDTR
```bash
git clone https://github.com/megvii-research/PETR.git
cd SDTR
mkdir ckpts
mkdir data
ln -s {mmdetection3d_path} ./mmdetection3d
ln -s {nuscenes_path} ./data/nuscenes
```




