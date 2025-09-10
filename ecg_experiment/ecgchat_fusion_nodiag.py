import os
import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from tqdm import tqdm

# -----------------------------
# Dataset
# -----------------------------
class PTBXLDataset(Dataset):
    def __init__(self, data_path, scp_path, base_path, train=True):
        self.data = pd.read_csv(data_path)
        self.scp = pd.read_csv(scp_path, index_col=0)
        self.base_path = base_path

        # binary split
        self.data = self.data[self.data['strat_fold'] < 9] if train else self.data[self.data['strat_fold'] == 9]

        # map to superclass
        self.data['superdiagnostic'] = self.data['scp_codes'].apply(self._map_superclass)

        # remove rows without labels
        self.data = self.data[self.data['superdiagnostic'].map(len) > 0]

        # build label set
        classes = sorted(set([item for sublist in self.data['superdiagnostic'] for item in sublist]))
        self.classes = classes
        print("Classes:", self.classes)

        # build label matrix
        self.data['label'] = self.data['superdiagnostic'].apply(lambda x: [1 if c in x else 0 for c in classes])

        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _map_superclass(self, scp_codes_str):
        import ast
        scp_codes = ast.literal_eval(scp_codes_str)
        res = []
        for code in scp_codes.keys():
            if code in self.scp.index:
                if self.scp.loc[code].diagnostic_class in ["NORM", "MI", "STTC", "HYP", "CD"]:
                    res.append(self.scp.loc[code].diagnostic_class)
        return list(set(res))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ECG signal
        record_path = os.path.join(self.base_path, row['filename_lr'])
        signal, _ = wfdb.rdsamp(record_path)
        signal = torch.tensor(signal.T, dtype=torch.float)  # shape (12, 1000)

        # text input (去掉诊断，只保留 lead info)
        leads_text = " ".join([f"lead {i}" for i in range(1, 13)])
        encoding = self.tokenizer(leads_text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        text_ids = encoding["input_ids"].squeeze(0)
        text_mask = encoding["attention_mask"].squeeze(0)

        label = torch.tensor(row['label'], dtype=torch.float)

        return signal, text_ids, text_mask, label

# -----------------------------
# Model
# -----------------------------
class ECGChatFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # CNN for ECG
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(64*128, 256),
            nn.ReLU()
        )
        # BERT for text
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(768, 256)

        # Fusion
        self.fc = nn.Linear(512, num_classes)

    def forward(self, signal, text_ids, text_mask):
        x_ecg = self.cnn(signal)
        x_text = self.text_model(input_ids=text_ids, attention_mask=text_mask).pooler_output
        x_text = self.text_fc(x_text)

        x = torch.cat([x_ecg, x_text], dim=1)
        out = self.fc(x)
        return out

# -----------------------------
# Training & Eval
# -----------------------------
def train_and_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
    scp_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv"
    base_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

    train_dataset = PTBXLDataset(data_path, scp_path, base_path, train=True)
    test_dataset = PTBXLDataset(data_path, scp_path, base_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = ECGChatFusion(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    for epoch in range(5):
        model.train()
        total_loss = 0
        for signals, text_ids, text_mask, labels in tqdm(train_loader):
            signals, text_ids, text_mask, labels = signals.to(device), text_ids.to(device), text_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals, text_ids, text_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for signals, text_ids, text_mask, labels in tqdm(test_loader):
            signals, text_ids, text_mask, labels = signals.to(device), text_ids.to(device), text_mask.to(device), labels.to(device)
            outputs = model(signals, text_ids, text_mask)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob, average="macro")
    except:
        auroc = float("nan")
    hloss = hamming_loss(y_true, y_pred)

    print(f"F1-macro: {f1_macro:.4f}, F1-micro: {f1_micro:.4f}, AUROC: {auroc:.4f}, Hamming Loss: {hloss:.4f}")

if __name__ == "__main__":
    train_and_eval()
