import torch
from torch.utils.data import DataLoader
from src.model import ECG_CNN
from src.dataset import PTBXLDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
import os

def train_model(data_path, label_path, epochs=10, batch_size=32, lr=0.001):
    dataset = PTBXLDataset(data_path, label_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ECG_CNN(num_classes=dataset.num_classes)
    model = model.cuda() if torch.cuda.is_available() else model

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs
            labels = labels.cuda() if torch.cuda.is_available() else labels

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_model.pth")
    print("Model saved to models/cnn_model.pth")