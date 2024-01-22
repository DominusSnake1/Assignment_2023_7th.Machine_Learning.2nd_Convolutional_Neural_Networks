from torchvision.models import resnet34
from torch import nn


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resnet34_model = resnet34(weights='DEFAULT')

        in_features = self.resnet34_model.fc.in_features
        self.resnet34_model.fc = nn.Linear(in_features, 7)

    def forward(self, x):
        return self.resnet34_model(x)
