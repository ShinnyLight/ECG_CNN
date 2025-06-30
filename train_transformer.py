# # train_transformer.py

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import os

# from src.dataset import PTBXLDataset
# from src.model_cnn_transformer import CNNTransformerECG

# DATA_PATH = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
# LABEL_CSV = os.path.join(DATA_PATH, "ptbxl_database.csv")
# MODEL_PATH = "models/transformer_model_71.pth"

# def train_model(data_path, label_path):
#     train_dataset = PTBXLDataset(data_path, label_path, train=True)
#     dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = CNNTransformerECG(num_classes=train_dataset.num_classes)
#     model.to(device)

#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     num_epochs = 10
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0.0

#         for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#         avg_loss = epoch_loss / len(dataloader)
#         print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

#     torch.save(model.state_dict(), MODEL_PATH)
#     print(f"✅ Transformer model saved to: {MODEL_PATH}")

# if __name__ == "__main__":
#     os.makedirs("models", exist_ok=True)
#     train_model(DATA_PATH, LABEL_CSV)
# train_transformer.py with pos_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.dataset import PTBXLDataset
from src.model_cnn_transformer import CNNTransformerECG

DATA_PATH = "D:/um/7023/brench dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
LABEL_CSV = os.path.join(DATA_PATH, "ptbxl_database.csv")
MODEL_PATH = "models/transformer_model_71.pth"
# MODEL_PATH = "models/transformer_model_71_weighted.pth"

def compute_pos_weights(dataset):
    labels = torch.stack([label for _, label in dataset])
    pos_counts = labels.sum(dim=0)
    neg_counts = labels.shape[0] - pos_counts
    weight = neg_counts / (pos_counts + 1e-5)
    return weight

def train_model(data_path, label_path):
    train_dataset = PTBXLDataset(data_path, label_path, train=True)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = CNNTransformerECG(num_classes=train_dataset.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Training on device: {next(model.parameters()).device}")

    pos_weight = compute_pos_weights(train_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Weighted Transformer model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_model(DATA_PATH, LABEL_CSV)
