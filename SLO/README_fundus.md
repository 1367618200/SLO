## 基本环境

#### 原始代码信息 (不用执行，仅方便版本比较和迁移)

```shell
git clone https://github.com/open-mmlab/mmclassification.git -c 1.x --branch v1.0.0rc5 -v GAMMA-mmcls
```

#### 下载及更新代码
```shell
mkdir ~/userfolder
cd ~/userfolder

# 下载代码
git clone https://github.com/CVIU-CSU/GAMMA-mmcls.git

# 从 github 更新本地代码
git fetch origin gamma
```

#### 环境配置

```shell
conda create -n gamma-mmcls python=3.7 -y
conda activate gamma-mmcls

# cuda 10.2 (e.g., 2080)
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch -y

# cuda 11 (e.g., 3080)
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -U openmim -i https://pypi.douban.com/simple/
mim install "mmcv>=2.0.0rc1" -i https://pypi.douban.com/simple/

pip install scikit-learn openpyxl -i https://pypi.douban.com/simple/

# for linux only
cd ~/userfolder/GAMMA-mmcls
chmod u+x tools/*
chmod u+x tools/*/*
mim install -e . -i https://pypi.douban.com/simple/
```

## 代码使用

#### 1. 数据预处理

```shell
# GAMMA数据集下载、解压
mkdir -p ~/userfolder/data
cd ~/userfolder/data

wget https://dataset-bj.cdn.bcebos.com/%E5%8C%BB%E7%96%97%E6%AF%94%E8%B5%9B/task1_gamma_grading.tar.gz.00
wget https://dataset-bj.cdn.bcebos.com/%E5%8C%BB%E7%96%97%E6%AF%94%E8%B5%9B/task1_gamma_grading.tar.gz.01
cat task1_gamma_grading.tar.gz* | tar -xzv
rm Glaucoma_grading/*/*/*/.DS_Store


cd ~/userfolder/GAMMA-mmcls

# 生成数据集的json标注文件
python tools/fundus/prepare_dataset.py

# 计算mean, std
python tools/fundus/mean_std.py
```

#### 2. 训练

```shell
cd ~/userfolder/GAMMA-mmcls
conda activate gamma-mmcls

# 单卡
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs_fundus/baseline/resnet34_b4_gamma.py
# 多卡
CUDA_VISIBLE_DEVICES=0,1 PORT=12345 bash tools/dist_train.sh configs_fundus/baseline/resnet34_b4_gamma.py 2
```

#### 3. 测试

```shell
cd ~/userfolder/GAMMA-mmcls
conda activate gamma-mmcls

# 在测试集上预测并生成csv （单卡/多卡）
# (for unlabeled dataset)
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs_fundus/baseline/resnet34_b4_gamma.py work_dirs/resnet34_b4_gamma/epoch_xxx.pth --dump-gamma-csv
CUDA_VISIBLE_DEVICES=0,1 PORT=12345 bash tools/dist_test.sh configs_fundus/baseline/resnet34_b4_gamma.py work_dirs/resnet34_b4_gamma/epoch_xxx.pth 2 --dump-gamma-csv

# 计算测试集的指标（单卡/多卡）
# (for labeled dataset) 测试集有标签的情况，配置文件中设置 test_dataloader = val_dataloader, test_evaluator = val_evaluator
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs_fundus/baseline/resnet34_b4_gamma.py work_dirs/resnet34_b4_gamma/epoch_xxx.pth
CUDA_VISIBLE_DEVICES=0,1 PORT=12345 bash tools/dist_test.sh configs_fundus/baseline/resnet34_b4_gamma.py work_dirs/resnet34_b4_gamma/epoch_xxx.pth 2
```

#### 4. 单元测试代码（供参考，非常有用）

参见目录 [tests/my](tests/my)，包含一些单元测试的代码，可用于调试、书写模块的代码。

- 数据集构建: [dataset.py](tests/my/dataset.py)
- 分类模型: [classifier_resnet34.py](tests/my/classifier_resnet34.py)
- 计算指标，生成csv: [metrics.py](tests/my/metrics.py)

## 附录

#### Anaconda 配置

```shell
wget -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.05-Linux-x86_64.sh
sh Anaconda3-2021.05-Linux-x86_64.sh
# 按照提示，init 选 yes

# zsh下初始化，适用于2080机器
echo 'export PATH=/root/anaconda3/bin:$PATH' >> ~/.zshrc
source ~/.zshrc
conda init zsh

# 设置conda镜像（加快conda命令的下载速度）
echo \
'channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.bfsu.edu.cn/anaconda
default_channels:
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/free
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/pro
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud
  msys2: https://mirrors.bfsu.edu.cn/anaconda/cloud
  bioconda: https://mirrors.bfsu.edu.cn/anaconda/cloud
  menpo: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.bfsu.edu.cn/anaconda/cloud
'> ~/.condarc
```
