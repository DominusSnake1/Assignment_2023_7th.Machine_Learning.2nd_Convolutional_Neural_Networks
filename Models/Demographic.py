from Models.ComplexCNN import ComplexCNN
from Models.SimpleCNN import SimpleCNN
from Models.ResNet import ResNet
import torch.nn as nn
import torch


class Demographic(nn.Module):
    def __init__(self, selected_model, demographic_feature_dim=128):
        super(Demographic, self).__init__()

        if selected_model == 'SimpleCNN':
            self.selected_model = SimpleCNN()
        elif selected_model == 'ComplexCNN':
            self.selected_model = ComplexCNN()
        elif selected_model == 'ResNet':
            self.selected_model = ResNet()
        else:
            raise ValueError("Invalid model selection. Choose 'simplecnn', 'complexcnn', or 'resnet'.")

        in_features = self.selected_model.fc.in_features
        self.selected_model.fc = nn.Identity()

        self.demographic_fc = nn.Linear(demographic_feature_dim, in_features)

        self.final_fc = nn.Linear(in_features + demographic_feature_dim, 7)

        self.relu = nn.ReLU()

    def forward(self, image, demographic_features):
        if isinstance(self.selected_model, nn.Module):
            image_features = self.selected_model(image)
        else:
            raise ValueError("Invalid model type. Expecting a PyTorch Module.")

        demographic_features = self.demographic_fc(demographic_features)
        demographic_features = self.relu(demographic_features)

        combined_features = torch.cat([image_features, demographic_features], dim=1)

        output = self.final_fc(combined_features)

        return output
