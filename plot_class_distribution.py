# plot_class_distribution.py (修正版)

import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
from collections import Counter

DATA_PATH = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
CSV_PATH = os.path.join(DATA_PATH, "ptbxl_database.csv")

def plot_distribution():
    df = pd.read_csv(CSV_PATH)

    # 解析字符串字典 scp_codes
    all_codes = []
    for s in df["scp_codes"]:
        try:
            code_dict = ast.literal_eval(s)
            all_codes.extend(code_dict.keys())
        except:
            continue

    # 统计标签出现次数
    counter = Counter(all_codes)
    sorted_counts = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))

    # 可视化前 50 个最常见标签
    plt.figure(figsize=(18, 6))
    plt.bar(sorted_counts.keys(), sorted_counts.values(), color='cornflowerblue')
    plt.title("PTB-XL Label Frequency Distribution (parsed from scp_codes)", fontsize=14)
    plt.xticks(rotation=90)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("label_distribution.png")
    print("✅ saved as label_distribution.png")

if __name__ == "__main__":
    plot_distribution()
