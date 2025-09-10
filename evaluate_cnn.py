import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
import pandas as pd
import os
from src.dataset import PTBXLDataset
from src.model import ECG_CNN

DATA_PATH = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
LABEL_CSV = os.path.join(DATA_PATH, "ptbxl_database.csv")
MODEL_PATH = "models/cnn_model.pth"

def evaluate_model(data_path, label_path, model_path):
    dataset = PTBXLDataset(data_path, label_path, train=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECG_CNN(num_classes=dataset.num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > 0.5).int()

            y_true.append(labels)
            y_pred.append(preds)
            y_prob.append(probs)

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    y_prob = torch.cat(y_prob).numpy()

    # === 统一指标 ===
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        auroc = float("nan")
    h_loss = hamming_loss(y_true, y_pred)

    print("\n📊 CNN Evaluation Results")
    print(f"F1-macro:   {f1_macro:.4f}")
    print(f"F1-micro:   {f1_micro:.4f}")
    print(f"AUROC:      {auroc:.4f}")
    print(f"Hamming Loss: {h_loss:.4f}")
    
class CNNBaseline(nn.Module):
    def __init__(self, num_classes):
        super(CNNBaseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

if __name__ == "__main__":
    evaluate_model(DATA_PATH, LABEL_CSV, MODEL_PATH)
