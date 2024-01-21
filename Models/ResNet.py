from torchvision.models import resnet34
from torchsummary import summary
from torch.optim import SGD
import numpy as np
import torch


class ResNet:
    def __init__(self):
        self.model = resnet34(weights='DEFAULT')
        self.model.fc = torch.nn.Linear(512, 7)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = SGD(
            self.model.parameters(),
            lr=0.001,
            momentum=0.9
        )
        self.device = 'cpu'

    def accuracy(self, output, target):
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).sum().item()
        total = target.size(0)
        return correct / total

    def train(self, epochs, train_loader, val_loader=None, print_period=10):
        self.model.to(self.device)
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            total_accuracy = 0.0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_accuracy += self.accuracy(outputs, labels)

                if i % print_period == print_period - 1:
                    avg_loss = running_loss / print_period
                    avg_accuracy = total_accuracy / print_period
                    print(f'[Epoch {epoch + 1}, Iteration {i + 1}] Average Loss: {avg_loss:.3f}, Average Accuracy: {avg_accuracy:.3f}')
                    running_loss = 0.0
                    total_accuracy = 0.0

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_accuracy = 0.0

                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        outputs = self.model(inputs)
                        val_loss += self.criterion(outputs, labels).item()
                        val_accuracy += self.accuracy(outputs, labels)

                avg_val_loss = val_loss / len(val_loader)
                avg_val_accuracy = val_accuracy / len(val_loader)
                print(f'\nValidation Loss: {avg_val_loss:.3f}, Validation Accuracy: {avg_val_accuracy:.3f}\n')

                self.model.train()

        print('Training finished\n')

    def test(self, test_loader):
        self.model.to(self.device)
        self.model.eval()

        test_loss = 0.0
        test_accuracy = 0.0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, labels).item()
                test_accuracy += self.accuracy(outputs, labels)

                predictions.append(outputs.argmax(dim=1).cpu().numpy())
                true_labels.append(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        avg_test_accuracy = test_accuracy / len(test_loader)

        print(f'Test Loss: {avg_test_loss:.3f}, Test Accuracy: {avg_test_accuracy:.3f}')

        predictions = np.concatenate(predictions)
        true_labels = np.concatenate(true_labels)

        self.model.train()

        return predictions, true_labels


def train_resnet(train_loader, test_loader, epochs):
    trainer = ResNet()
    summary(trainer.model, (3, 224, 224))

    trainer.train(epochs=epochs, train_loader=train_loader, val_loader=test_loader)
    predictions, true_labels = trainer.test(test_loader)

    return predictions, true_labels
