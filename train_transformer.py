import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import wfdb
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

# =========================
# Dataset
# =========================
class PTBXLDataset(Dataset):
    def __init__(self, data_path, scp_path, base_path, train=True, folds=[1,2,3,4,5]):
        self.data = pd.read_csv(data_path)
        self.scp_statements = pd.read_csv(scp_path, index_col=0)

        # 取 diagnostic class 信息
        self.scp_codes = self.scp_statements[self.scp_statements.diagnostic == 1]
        self.data['superdiagnostic'] = self.data['scp_codes'].apply(self._aggregate_diagnostic)

        # train/test split
        if train:
            self.data = self.data[self.data.strat_fold.isin(folds[:-1])]
        else:
            self.data = self.data[self.data.strat_fold == folds[-1]]

        self.base_path = base_path

        # binarizer for multilabel
        self.mlb = MultiLabelBinarizer(classes=sorted(set([c for l in self.data['superdiagnostic'] for c in l])))
        self.mlb.fit(self.data['superdiagnostic'])

        print("Classes:", self.mlb.classes_)

    def _aggregate_diagnostic(self, scp_codes_str):
        scp_codes = eval(scp_codes_str)
        return [self.scp_statements.loc[k].diagnostic_class for k in scp_codes.keys()
                if k in self.scp_statements.index and self.scp_statements.loc[k].diagnostic == 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        record_path = os.path.join(self.base_path, row.filename_lr)
        signal, _ = wfdb.rdsamp(record_path)
        signal = torch.tensor(signal.T, dtype=torch.float)  # shape [12, 1000]
        label = self.mlb.transform([row.superdiagnostic])[0]
        return signal, torch.tensor(label, dtype=torch.float)

# =========================
# Transformer 模型
# =========================
class TransformerECG(nn.Module):
    def __init__(self, num_classes, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Conv1d(12, d_model, kernel_size=1)  # [B,12,1000] → [B,128,1000]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)  # [B,128,1000]
        x = x.permute(0, 2, 1)  # [B,1000,128]
        x = self.encoder(x)     # [B,1000,128]
        x = x.mean(dim=1)       # pool
        return self.fc(x)

# =========================
# Focal Loss (可选)
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # pt = 1 - p
        loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

# =========================
# Train & Eval
# =========================
def train_and_eval(use_focal=False, lr=1e-4, batch_size=32, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    data_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
    scp_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv"
    base_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"

    train_dataset = PTBXLDataset(data_path, scp_path, base_path, train=True)
    test_dataset = PTBXLDataset(data_path, scp_path, base_path, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    num_classes = len(train_dataset.mlb.classes_)
    model = TransformerECG(num_classes=num_classes).to(device)

    # pos_weight 计算
    label_matrix = np.vstack([train_dataset.mlb.transform([row]) for row in train_dataset.data['superdiagnostic']])
    pos_counts = label_matrix.sum(axis=0)
    neg_counts = len(train_dataset) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float, device=device)

    # loss
    if use_focal:
        criterion = FocalLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # eval
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Eval"):
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    f1_macro = f1_score(all_labels, all_preds>0.5, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds>0.5, average='micro', zero_division=0)
    try:
        auroc = roc_auc_score(all_labels, all_preds, average='macro')
    except:
        auroc = float('nan')
    hloss = hamming_loss(all_labels, all_preds>0.5)

    print(f"F1-macro: {f1_macro:.4f}, F1-micro: {f1_micro:.4f}, AUROC: {auroc:.4f}, Hamming Loss: {hloss:.4f}")

    torch.save(model.state_dict(), "models/transformer_model_balanced.pth")
    print("✅ Transformer model saved.")

if __name__ == "__main__":
    # 改 use_focal=True 就能启用 Focal Loss
    train_and_eval(use_focal=False)
