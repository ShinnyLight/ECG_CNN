import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wfdb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# ======================
# 数据路径
# ======================
ptbxl_root = r"D:\um\7023\brench dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
data_path = os.path.join(ptbxl_root, "ptbxl_database.csv")
scp_path = os.path.join(ptbxl_root, "scp_statements.csv")

# ======================
# 加载 CSV
# ======================
df = pd.read_csv(data_path)
scp_df = pd.read_csv(scp_path, index_col=0)

# 只保留 diagnostic_class 列
df['diagnostic_superclass'] = df['scp_codes'].apply(
    lambda x: [k for k in eval(x).keys() if k in scp_df.index and scp_df.loc[k].diagnostic_class not in ["", None]]
)
df = df[df['diagnostic_superclass'].map(len) > 0]  # 去掉空标签
df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(lambda x: scp_df.loc[x[0]].diagnostic_class)

classes = df['diagnostic_superclass'].unique().tolist()
class_to_idx = {c: i for i, c in enumerate(classes)}
df['label'] = df['diagnostic_superclass'].map(class_to_idx)

print(f"Classes: {classes}")

# ======================
# Dataset 类
# ======================
class PTBXL_Dataset(Dataset):
    def __init__(self, df, root_dir):
        self.df = df
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record_path = os.path.join(self.root_dir, self.df.iloc[idx].filename_lr)
        signal, _ = wfdb.rdsamp(record_path)
        label = self.df.iloc[idx].label
        return torch.tensor(signal.T, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ======================
# 简单模型 (1D CNN)
# ======================
class SimpleECGNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ======================
# 训练 & 评估
# ======================
dataset = PTBXL_Dataset(df, ptbxl_root)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleECGNet(len(classes)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(3):
    model.train()
    total_loss = 0
    for signals, labels in tqdm(dataloader):
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# 测试
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for signals, labels in dataloader:
        signals, labels = signals.to(device), labels.to(device)
        outputs = model(signals)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred, average="macro"))
