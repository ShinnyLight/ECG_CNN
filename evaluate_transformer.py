# evaluate_transformer.py - Transformer evaluation (sample-wise + per-class visualization + dual accuracy)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.dataset import PTBXLDataset
from src.model_cnn_transformer import CNNTransformerECG  # 请确保你 transformer 模型类名为 CNNTransformerECG

# === 设置路径 ===
DATA_PATH = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
LABEL_CSV = os.path.join(DATA_PATH, "ptbxl_database.csv")
MODEL_PATH = "models/transformer_model_71.pth"
SAVE_CSV = "results/transformer_predictions.csv"
SAVE_IMG = "results/transformer_f1_per_class.png"

def evaluate_model(data_path, label_path, model_path):
    dataset = PTBXLDataset(data_path, label_path, train=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNTransformerECG(num_classes=dataset.num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().tolist()

            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    df_out = pd.DataFrame(y_pred, columns=[f"Pred_{i}" for i in range(dataset.num_classes)])
    os.makedirs("results", exist_ok=True)
    df_out.to_csv(SAVE_CSV, index=False)
    print(f"\n✅ Predictions saved to {SAVE_CSV}")

    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    print("\n📊 Transformer Sample-wise Evaluation Results:")
    sample_precision = precision_score(y_true_tensor, y_pred_tensor, average='samples', zero_division=0)
    sample_recall    = recall_score(y_true_tensor, y_pred_tensor, average='samples', zero_division=0)
    sample_f1        = f1_score(y_true_tensor, y_pred_tensor, average='samples', zero_division=0)
    strict_accuracy  = (y_true_tensor == y_pred_tensor).all(dim=1).float().mean().item()
    loose_accuracy   = (y_true_tensor == y_pred_tensor).float().mean().item()

    print(f"Strict Accuracy (Exact Match):   {strict_accuracy:.4f}")
    print(f"Per-label Accuracy (Avg Match):  {loose_accuracy:.4f}")
    print(f"Precision (sample-wise):         {sample_precision:.4f}")
    print(f"Recall (sample-wise):            {sample_recall:.4f}")
    print(f"F1 Score (sample-wise):          {sample_f1:.4f}")

    print("\n📉 Generating per-class F1-score chart...")
    report = classification_report(y_true_tensor, y_pred_tensor, output_dict=True, zero_division=0)

    class_f1 = {}
    for key in report:
        if key.isdigit():
            class_f1[key] = report[key]['f1-score']

    class_f1_sorted = dict(sorted(class_f1.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(18, 6))
    plt.bar(class_f1_sorted.keys(), class_f1_sorted.values(), color='mediumseagreen')
    plt.xticks(rotation=90)
    plt.title("Per-Class F1 Score (Transformer)")
    plt.xlabel("Label Index")
    plt.ylabel("F1 Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(SAVE_IMG)
    print(f"✅ Per-class F1-score 图已保存为 {SAVE_IMG}")

if __name__ == "__main__":
    evaluate_model(DATA_PATH, LABEL_CSV, MODEL_PATH)

