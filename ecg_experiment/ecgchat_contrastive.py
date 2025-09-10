import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wfdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from tqdm import tqdm

# =====================
# Dataset
# =====================
class PTBXLDataset(Dataset):
    def __init__(self, data_path, scp_path, base_path, train=True, split=0.8):
        self.base_path = base_path
        self.data = pd.read_csv(data_path)
        self.scp_statements = pd.read_csv(scp_path, index_col=0)

        
        self.data['superdiagnostic'] = self.data['scp_codes'].apply(self._aggregate_diagnostic)
        self.data = self.data[self.data['superdiagnostic'].map(lambda x: len(x) > 0)]

        
        all_labels = []
        for sublist in self.data['superdiagnostic']:
            for item in sublist:
                if isinstance(item, str):
                    all_labels.append(item)

        classes = sorted(set(all_labels))
        self.class_map = {c: i for i, c in enumerate(classes)}

        # 转换成 one-hot 多标签
        self.data['label'] = self.data['superdiagnostic'].map(
            lambda x: np.array([1 if c in x else 0 for c in self.class_map])
        )

        # 划分训练 / 测试
        split_idx = int(len(self.data) * split)
        self.data = self.data.iloc[:split_idx] if train else self.data.iloc[split_idx:]

    def _aggregate_diagnostic(self, scp_codes):
        scp_codes = eval(scp_codes)
        return [
            self.scp_statements.loc[k].diagnostic_class
            for k in scp_codes.keys()
            if k in self.scp_statements.index
            and isinstance(self.scp_statements.loc[k].diagnostic_class, str)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        record_path = os.path.join(self.base_path, row['filename_lr'])
        signal, _ = wfdb.rdsamp(record_path)
        signal = torch.tensor(signal.T, dtype=torch.float32)
        label = torch.tensor(row['label'], dtype=torch.float32)
        return signal, label


# =====================
# Model
# =====================
class ECGChatMini(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ecg_encoder = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.proj_ecg = nn.Linear(128, 128)
        self.proj_text = nn.Linear(768, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, ecg, text_input_ids, text_attention_mask):
        ecg_feat = self.ecg_encoder(ecg)
        ecg_proj = self.proj_ecg(ecg_feat)

        text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_feat = text_outputs.pooler_output
        text_proj = self.proj_text(text_feat)

        logits = self.classifier(ecg_proj)
        return logits, ecg_proj, text_proj

# =====================
# Training & Evaluation
# =====================
def train_and_eval():
    data_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
    scp_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv"
    base_path = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"

    train_dataset = PTBXLDataset(data_path, scp_path, base_path, train=True)
    test_dataset = PTBXLDataset(data_path, scp_path, base_path, train=False, split=0.8)

    num_classes = len(train_dataset.class_map)
    print("Classes:", list(train_dataset.class_map.keys()))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGChatMini(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for epoch in range(5):  # run 5 epochs
        model.train()
        total_loss = 0
        for signals, labels in tqdm(train_loader):
            signals, labels = signals.to(device), labels.to(device)

            # dummy text inputs
            text = ["ECG record"] * signals.size(0)
            encodings = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)

            optimizer.zero_grad()
            logits, ecg_proj, text_proj = model(signals, encodings['input_ids'], encodings['attention_mask'])

            loss_cls = criterion(logits, labels)

            loss = loss_cls
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for signals, labels in tqdm(test_loader):
            signals, labels = signals.to(device), labels.to(device)

            text = ["ECG record"] * signals.size(0)
            encodings = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)

            logits, _, _ = model(signals, encodings['input_ids'], encodings['attention_mask'])
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)

    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    try:
        auroc = roc_auc_score(all_labels, all_probs, average="macro")
    except:
        auroc = float('nan')
    ham_loss = hamming_loss(all_labels, all_preds)

    print(f"F1-macro: {f1_macro:.4f}, F1-micro: {f1_micro:.4f}, AUROC: {auroc:.4f}, Hamming Loss: {ham_loss:.4f}")


if __name__ == "__main__":
    train_and_eval()
