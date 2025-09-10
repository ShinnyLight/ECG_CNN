import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import wfdb
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from tqdm import tqdm

# ================== Dataset ==================
class PTBXLDataset(Dataset):
    def __init__(self, data_path, scp_path, base_path, train=True, max_len=64):
        self.base_path = base_path
        self.data = pd.read_csv(data_path)
        scp_df = pd.read_csv(scp_path, index_col=0)

        # Keep diagnostic superclass only
        self.data = self.data[self.data['strat_fold'] <= 8] if train else self.data[self.data['strat_fold'] > 8]
        self.data = self.data.reset_index(drop=True)

        # Map scp_codes to diagnostic superclasses
        self.data['superdiagnostic'] = self.data['scp_codes'].apply(
            lambda x: [scp_df.loc[k].diagnostic_class for k in eval(x).keys() if k in scp_df.index and pd.notna(scp_df.loc[k].diagnostic_class)]
        )

        # Build class set
        classes = sorted(set([item for sublist in self.data['superdiagnostic'] for item in sublist]))
        self.classes = [c for c in classes if c in ["NORM", "MI", "STTC", "HYP", "CD"]]
        print("Classes:", self.classes)

        # Binarize labels
        self.data['label'] = self.data['superdiagnostic'].apply(
            lambda x: [1 if c in x else 0 for c in self.classes]
        )

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ECG
        record_path = os.path.join(self.base_path, row['filename_lr'])
        signal, _ = wfdb.rdsamp(record_path)
        signal = torch.tensor(signal.T, dtype=torch.float)  # shape [12, len]
        signal = nn.functional.interpolate(signal.unsqueeze(0), size=5000, mode="linear").squeeze(0)  # normalize length

        # Text (use scp_codes keys as description)
        text_input = " ".join(eval(row['scp_codes']).keys())
        text_enc = self.tokenizer(text_input, padding="max_length", truncation=True,
                                  max_length=self.max_len, return_tensors="pt")

        label = torch.tensor(row['label'], dtype=torch.float)

        return signal, text_enc["input_ids"].squeeze(0), text_enc["attention_mask"].squeeze(0), label

# ================== Model ==================
class ECGChatMini(nn.Module):
    def __init__(self, num_classes, text_hidden=768, ecg_hidden=128, fusion_hidden=256):
        super().__init__()
        # ECG encoder (1D CNN)
        self.ecg_encoder = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, ecg_hidden, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(ecg_hidden + text_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, ecg, input_ids, attention_mask):
        ecg_feat = self.ecg_encoder(ecg).squeeze(-1)  # [B, ecg_hidden]
        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # [B, text_hidden]
        fused = torch.cat([ecg_feat, text_feat], dim=1)
        return self.fc(fused)

# ================== Training ==================
def train_and_eval():
    data_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
    scp_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv"
    base_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"

    train_dataset = PTBXLDataset(data_path, scp_path, base_path, train=True)
    test_dataset = PTBXLDataset(data_path, scp_path, base_path, train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGChatMini(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for ecg, input_ids, attention_mask, labels in tqdm(train_loader):
            ecg, input_ids, attention_mask, labels = ecg.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(ecg, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for ecg, input_ids, attention_mask, labels in tqdm(test_loader):
            ecg, input_ids, attention_mask, labels = ecg.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(ecg, input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    y_true, y_pred, y_score = np.array(y_true), np.array(y_pred), np.array(y_score)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    auroc = roc_auc_score(y_true, y_score, average="macro")
    hamming = hamming_loss(y_true, y_pred)

    print(f"F1-macro: {f1_macro:.4f}, F1-micro: {f1_micro:.4f}, AUROC: {auroc:.4f}, Hamming Loss: {hamming:.4f}")

if __name__ == "__main__":
    train_and_eval()
