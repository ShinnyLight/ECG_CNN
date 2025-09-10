import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wfdb
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from tqdm import tqdm
from transformers import BertModel

# ======================
# Dataset
# ======================
class PTBXLDataset(Dataset):
    def __init__(self, data_path, scp_path, base_path, train=True, fold=10):
        self.data = pd.read_csv(data_path)
        self.scp_df = pd.read_csv(scp_path, index_col=0)

        # superdiagnostic classes (5)
        self.scp_df = self.scp_df[self.scp_df.diagnostic_class.notnull()]
        self.data['superdiagnostic'] = self.data['scp_codes'].apply(
            lambda x: self._aggregate_superdiagnostic(eval(x))
        )

        # 只保留有标签的
        self.data = self.data[self.data['superdiagnostic'].map(len) > 0]

        # train/test split
        if train:
            self.data = self.data[self.data.strat_fold != fold]
        else:
            self.data = self.data[self.data.strat_fold == fold]

        # 取所有出现的类
        classes = sorted(set([item for sublist in self.data['superdiagnostic'] for item in sublist]))
        self.classes = classes
        self.class_map = {c: i for i, c in enumerate(classes)}
        print("Classes:", classes)

        self.base_path = base_path

    def _aggregate_superdiagnostic(self, scp_codes):
        labels = []
        for k in scp_codes.keys():
            if k in self.scp_df.index:
                diag = self.scp_df.loc[k].diagnostic_class
                if diag in ["NORM", "MI", "STTC", "HYP", "CD"]:
                    labels.append(diag)
        return list(set(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        record = row['filename_lr']
        record_path = os.path.join(self.base_path, record)

        signal, _ = wfdb.rdsamp(record_path)
        signal = torch.tensor(signal.T, dtype=torch.float32)  # (12, 1000)

        labels = torch.zeros(len(self.classes))
        for c in row['superdiagnostic']:
            labels[self.class_map[c]] = 1.0

        return signal, labels

# ======================
# Models
# ======================
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        x = self.conv(x)  # (batch, 64, 1)
        return x.view(x.size(0), -1)  # (batch, 64)

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        self.ecg_encoder = CNNEncoder()
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(64 + 768, num_classes)

    def forward(self, ecg, text_inputs=None):
        ecg_feat = self.ecg_encoder(ecg)
        # 用 [CLS] embedding 作为 text embedding (这里不输入诊断标签 → 用零向量代替)
        batch_size = ecg.size(0)
        text_feat = torch.zeros(batch_size, 768, device=ecg.device)
        combined = torch.cat([ecg_feat, text_feat], dim=1)
        return self.fc(combined)

# ======================
# Training / Evaluation
# ======================
def train_and_eval(use_focal=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
    scp_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv"
    base_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"

    train_dataset = PTBXLDataset(data_path, scp_path, base_path, train=True)
    test_dataset = PTBXLDataset(data_path, scp_path, base_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = len(train_dataset.classes)
    model = FusionModel(num_classes).to(device)

    # pos_weight
    label_counts = np.zeros(num_classes)
    for _, labels in train_dataset:
        label_counts += labels.numpy()
    pos_weight = torch.tensor((len(train_dataset) - label_counts) / (label_counts + 1e-5), dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ====== Train ======
    for epoch in range(5):
        model.train()
        total_loss = 0
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # ====== Eval ======
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Eval"):
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            probs = torch.sigmoid(outputs)
            y_true.append(labels.cpu().numpy())
            y_pred.append((probs > 0.5).int().cpu().numpy())
            y_score.append(probs.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_score = np.vstack(y_score)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_score, average="macro")
    except ValueError:
        auroc = float("nan")
    h_loss = hamming_loss(y_true, y_pred)

    print(f"F1-macro: {f1_macro:.4f}, F1-micro: {f1_micro:.4f}, AUROC: {auroc:.4f}, Hamming Loss: {h_loss:.4f}")

if __name__ == "__main__":
    train_and_eval()
