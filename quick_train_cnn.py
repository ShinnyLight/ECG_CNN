# quick_train_cnn.py
# 👀 快速调试 CNN 学习能力（小样本+低轮数）

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

from src.dataset import PTBXLDataset
from src.model import ECG_CNN

DATA_PATH = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
LABEL_CSV = os.path.join(DATA_PATH, "ptbxl_database.csv")
MODEL_PATH = "models/cnn_quick_debug.pth"

def compute_pos_weights(dataset):
    labels = torch.stack([label for _, label in dataset])
    pos_counts = labels.sum(dim=0)
    neg_counts = labels.shape[0] - pos_counts
    weight = neg_counts / (pos_counts + 1e-5)
    return weight

def train_small_sample(data_path, label_path):
    print("🚀 Starting quick small-batch training...")

    # 原始数据
    full_dataset = PTBXLDataset(data_path, label_path, train=True)

    # 👇 只取前 500 条样本
    small_dataset = Subset(full_dataset, range(500))
    dataloader = DataLoader(small_dataset, batch_size=32, shuffle=True)

    # 初始化模型和设备
    model = ECG_CNN(num_classes=full_dataset.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Training on device: {next(model.parameters()).device}")

    # 加权 BCE Loss
    pos_weight = compute_pos_weights(small_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 🚀 快速训练 2 轮
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}")

    # ✅ 模型保存
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Debug CNN model saved to: {MODEL_PATH}")

    # 🔍 检查输出
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = torch.sigmoid(model(inputs))
            print("📊 Sample output mean prob:", outputs.mean().item())
            break

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_small_sample(DATA_PATH, LABEL_CSV)
