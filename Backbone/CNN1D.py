import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()

        # -------- FEATURE EXTRACTOR --------
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # output: (B, 64, 1)

        # -------- FC-ADAPT (Layer for Domain Alignment) --------
        self.fc_adapt = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # -------- FINAL CLASSIFIER --------
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 1, L)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)          # (B, 64, 1)
        x = x.squeeze(-1)            # (B, 64)

        features = self.fc_adapt(x)  # (B, 128)
        logits = self.classifier(features)

        return logits, features
