# CNN Baseline for PTB-XL ECG Classification

## 1. 环境准备

推荐使用 conda 环境（Python 3.10）并安装依赖：

```
conda create -n ecgcnn python=3.10
conda activate ecgcnn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install wfdb pandas scikit-learn tqdm matplotlib
```

## 2. 项目结构

```
ECG_CNN/
├── src/
│   ├── dataset.py         ← 数据预处理
│   ├── model.py           ← CNN 模型结构
│   ├── train.py           ← 训练主脚本
├── train_cnn.py           ← 运行入口（修改路径）
```

## 3. 运行方法

编辑 `train_cnn.py` 设置你的数据路径，然后运行：

```
python train_cnn.py
```

模型将保存在 `models/cnn_model.pth`