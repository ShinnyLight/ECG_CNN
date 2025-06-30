# src/dataset.py

import os
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class PTBXLDataset(Dataset):
    def __init__(self, data_path, label_path, train=True, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.label_path = label_path
        self.train = train

        df = pd.read_csv(label_path)
        df = df[df['scp_codes'].notnull()]  # drop missing

        # 解析多标签列为独热向量
        def parse_labels(s):
            labels = eval(s)
            return {k: 1 for k in labels.keys()}

        labels_list = df['scp_codes'].map(parse_labels)
        all_classes = sorted({label for labels in labels_list for label in labels})
        self.classes = all_classes
        self.num_classes = len(self.classes)

        def encode_labels(labels):
            encoded = np.zeros(len(self.classes))
            for label in labels:
                if label in self.classes:
                    encoded[self.classes.index(label)] = 1
            return encoded

        df['encoded_labels'] = labels_list.map(lambda d: encode_labels(d.keys()))

        # 数据划分
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        self.df = train_df if train else test_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_path = os.path.join(self.data_path, row['filename_hr'])  # ECG信号路径
        signal, _ = wfdb.rdsamp(rec_path)
        signal = torch.tensor(signal, dtype=torch.float32).T  # shape: [leads, length]
        label = torch.tensor(row['encoded_labels'], dtype=torch.float32)
        return signal, label
