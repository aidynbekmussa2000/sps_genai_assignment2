import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),  # 64→32
            nn.Conv2d(16,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2)  # 32→16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*16*16, 100), nn.ReLU(),
            nn.Linear(100, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)