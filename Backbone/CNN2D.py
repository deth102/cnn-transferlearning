import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, num_classes):
        super(CNN2D, self).__init__()

        # -------- FEATURE EXTRACTOR --------
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Lưu ý: STFT input shape F x T sẽ quyết định kích thước flatten
        # nhưng MaxPool2d(2) ba lần ≈ chia 8 mỗi chiều
        self.gap = nn.AdaptiveAvgPool2d((1, 1))   # output: (B, 64, 1, 1)

        # -------- FC-ADAPT FOR DOMAIN ALIGNMENT --------
        self.fc_adapt = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # -------- CLASSIFIER --------
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B,1,F,T)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.gap(x)                 # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)       # (B, 64)

        features = self.fc_adapt(x)     # (B, 128)
        logits = self.classifier(features)

        return logits, features
