import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb
from sklearn.metrics import f1_score, roc_curve, auc, multilabel_confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast

# ======================
# CNN Baseline
# ======================
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


# ======================
# Dataset Loader (修复 diagnostic_superclass)
# ======================
class PTBXLDataset(Dataset):
    def __init__(self, data_path, scp_path, base_path, train=True):
        self.data = pd.read_csv(data_path)
        self.scp = pd.read_csv(scp_path, index_col=0)

        # 只保留有 diagnostic_class 的项
        self.scp = self.scp[self.scp['diagnostic_class'].notna()]

        # 映射 diagnostic_superclass
        def map_superclass(scp_codes_str):
            try:
                scp_codes = ast.literal_eval(scp_codes_str)
            except Exception:
                return None
            superclasses = set()
            for code in scp_codes.keys():
                if code in self.scp.index:
                    superclass = self.scp.loc[code, 'diagnostic_class']
                    if superclass in ["NORM", "MI", "STTC", "HYP", "CD"]:
                        superclasses.add(superclass)
            return list(superclasses) if superclasses else None

        self.data["superdiagnostic"] = self.data["scp_codes"].apply(map_superclass)
        self.data = self.data.dropna(subset=["superdiagnostic"])

        # train/test split
        self.data = self.data[self.data.strat_fold < 9] if train else self.data[self.data.strat_fold == 10]

        # multi-label binarization
        self.classes = ["CD", "HYP", "MI", "NORM", "STTC"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.base_path = base_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        record_path = os.path.join(self.base_path, row.filename_lr)
        signal, _ = wfdb.rdsamp(record_path)
        signal = torch.tensor(signal.T, dtype=torch.float32)

        label = torch.zeros(len(self.classes))
        for c in row["superdiagnostic"]:
            if c in self.class_to_idx:
                label[self.class_to_idx[c]] = 1.0
        return signal, label


def get_dataloaders(data_path, scp_path, base_path, batch_size=32):
    train_dataset = PTBXLDataset(data_path, scp_path, base_path, train=True)
    test_dataset = PTBXLDataset(data_path, scp_path, base_path, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, len(train_dataset.classes), train_dataset.classes


# ======================
# Evaluation + Plots
# ======================
def evaluate_and_plot():
    data_path = r"D:\um\7023\brench dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptbxl_database.csv"
    scp_path = r"D:\um\7023\brench dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\scp_statements.csv"
    base_path = r"D:\um\7023\brench dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

    model_path = r"D:\um\7023\ECG_CNN\models\cnn_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader, num_classes, classes = get_dataloaders(data_path, scp_path, base_path, batch_size=32)
    print("✅ Classes:", classes)

    model = CNNBaseline(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Evaluating"):
            signals, labels = signals.to(device), labels.to(device)
            outputs = torch.sigmoid(model(signals))
            preds = (outputs > 0.5).float()

            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_score.append(outputs.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_score = np.vstack(y_score)

    # Confusion Matrix
    cm = multilabel_confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=0)
    fig, ax = plt.subplots()
    im = ax.imshow(cm_sum, cmap="Blues")
    ax.set_title("Aggregated Confusion Matrix")
    plt.colorbar(im, ax=ax)
    plt.savefig("cnn_confusion_matrix.png")
    plt.close()

    # Per-class F1
    f1_scores = f1_score(y_true, y_pred, average=None)
    plt.figure(figsize=(8, 5))
    plt.bar(classes, f1_scores)
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores")
    plt.savefig("cnn_f1_per_class.png")
    plt.close()

    # ROC Curves
    plt.figure(figsize=(7, 6))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{c} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (5 Classes)")
    plt.legend()
    plt.savefig("cnn_roc_curves.png")
    plt.close()

    # Class Distribution
    counts = np.sum(y_true, axis=0)
    plt.figure(figsize=(8, 5))
    plt.bar(classes, counts)
    plt.ylabel("Count")
    plt.title("Test Set Class Distribution")
    plt.savefig("cnn_class_distribution.png")
    plt.close()

    print("✅ All 4 plots saved!")


if __name__ == "__main__":
    evaluate_and_plot()
