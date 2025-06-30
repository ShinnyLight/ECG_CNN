# import os
# import torch
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# # import matplotlib.pyplot as plt
# import numpy as np
# from torch.utils.data import DataLoader
# from src.dataset import PTBXLDataset
# from src.model import ECG_CNN

# DATA_PATH = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
# LABEL_CSV = os.path.join(DATA_PATH, "ptbxl_database.csv")
# MODEL_PATH = "models/cnn_model_71.pth"
# # MODEL_PATH = "models/cnn_model_71_weighted.pth"
# def evaluate_model(data_path, label_path, model_path):
#     # 加载测试集
#     test_dataset = PTBXLDataset(data_path, label_path, train=False)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#     # 加载模型
#     model = ECG_CNN(num_classes=test_dataset.num_classes)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     model.to("cuda" if torch.cuda.is_available() else "cpu")

#     y_true, y_pred = [], []

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
#             outputs = model(inputs).cpu()

#             y_true.extend(labels.numpy())
#             y_pred.extend((outputs > 0.5).int().numpy())

#     # 转换为 DataFrame
#     df_true = pd.DataFrame(y_true, columns=[f"True_{cls}" for cls in test_dataset.classes])
#     df_pred = pd.DataFrame(y_pred, columns=[f"Pred_{cls}" for cls in test_dataset.classes])
#     df_out = pd.concat([df_true, df_pred], axis=1)

#     # 保存输出
#     df_out.to_csv("results/cnn_predictions.csv", index=False)
#     print("✅ Predictions saved to results/cnn_predictions.csv")

#     # 多标签平均指标
#     y_true_tensor = torch.tensor(y_true)
#     y_pred_tensor = torch.tensor(y_pred)
#     accuracy = (y_true_tensor == y_pred_tensor).float().mean().item()
#     precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
#     f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

#     print("\n📊 Evaluation Results:")
#     print(f"Accuracy:  {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall:    {recall:.4f}")
#     print(f"F1 Score:  {f1:.4f}")

# if __name__ == "__main__":
#     os.makedirs("results", exist_ok=True)
#     evaluate_model(DATA_PATH, LABEL_CSV, MODEL_PATH)
# evaluate_cnn.py - upgraded version with visualization

# evaluate_cnn.py - CNN baseline evaluation (sample-wise + per-class visualization)

# evaluate_cnn.py - CNN baseline evaluation (sample-wise + per-class visualization + dual accuracy)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.dataset import PTBXLDataset
from src.model import ECG_CNN

# === 设置路径 ===
DATA_PATH = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
LABEL_CSV = os.path.join(DATA_PATH, "ptbxl_database.csv")
MODEL_PATH = "models/cnn_model_71.pth"  # ← 替换为未加权模型
SAVE_CSV = "results/cnn_predictions.csv"
SAVE_IMG = "results/cnn_f1_per_class.png"

def evaluate_model(data_path, label_path, model_path):
    # 加载数据集
    dataset = PTBXLDataset(data_path, label_path, train=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = ECG_CNN(num_classes=dataset.num_classes)
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

    # 保存预测结果
    df_out = pd.DataFrame(y_pred, columns=[f"Pred_{i}" for i in range(dataset.num_classes)])
    os.makedirs("results", exist_ok=True)
    df_out.to_csv(SAVE_CSV, index=False)
    print(f"\n✅ Predictions saved to {SAVE_CSV}")

    # === 上半部分：Sample-wise 性能（表格用） ===
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    print("\n📊 CNN Sample-wise Evaluation Results:")
    sample_precision = precision_score(y_true_tensor, y_pred_tensor, average='samples', zero_division=0)
    sample_recall    = recall_score(y_true_tensor, y_pred_tensor, average='samples', zero_division=0)
    sample_f1        = f1_score(y_true_tensor, y_pred_tensor, average='samples', zero_division=0)

    # ✅ 两种 Accuracy
    strict_accuracy = (y_true_tensor == y_pred_tensor).all(dim=1).float().mean().item()
    loose_accuracy  = (y_true_tensor == y_pred_tensor).float().mean().item()

    print(f"Strict Accuracy (Exact Match):   {strict_accuracy:.4f}")
    print(f"Per-label Accuracy (Avg Match):  {loose_accuracy:.4f}")
    print(f"Precision (sample-wise):         {sample_precision:.4f}")
    print(f"Recall (sample-wise):            {sample_recall:.4f}")
    print(f"F1 Score (sample-wise):          {sample_f1:.4f}")

    # === 下半部分：Per-class 可视化（图像用） ===
    print("\n📉 Generating per-class F1-score chart...")
    report = classification_report(y_true_tensor, y_pred_tensor, output_dict=True, zero_division=0)

    class_f1 = {}
    for key in report:
        if key.isdigit():
            class_f1[key] = report[key]['f1-score']

    class_f1_sorted = dict(sorted(class_f1.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(18, 6))
    plt.bar(class_f1_sorted.keys(), class_f1_sorted.values(), color='salmon')
    plt.xticks(rotation=90)
    plt.title("Per-Class F1 Score (CNN)")
    plt.xlabel("Label Index")
    plt.ylabel("F1 Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(SAVE_IMG)
    print(f"✅ Per-class F1-score 图已保存为 {SAVE_IMG}")

if __name__ == "__main__":
    evaluate_model(DATA_PATH, LABEL_CSV, MODEL_PATH)
