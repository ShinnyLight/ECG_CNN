import torch
import torch.nn as nn

class CNNTransformerECG(nn.Module):
    def __init__(self, num_classes, input_length=5000):
        super(CNNTransformerECG, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [32, 2500]
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [64, 1250]
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [128, 625]
        )

        conv_out_dim = 625
        embed_dim = 128
        self.pos_encoder = nn.Identity()  # 可选：加位置编码

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)  # [B, C=128, T=625]
        x = x.permute(0, 2, 1)  # [B, T=625, C=128]
        x = self.pos_encoder(x)
        x = self.transformer(x)  # [B, T, C]
        x = x.contiguous().view(x.size(0), -1)  # Flatten
        out = self.classifier(x)
        return out
