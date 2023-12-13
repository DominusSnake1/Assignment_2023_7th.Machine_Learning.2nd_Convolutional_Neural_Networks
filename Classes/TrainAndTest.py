import torch
import torch.nn as nn
import torch.optim as optim
from Classes.Transformations import Transformations


class TrainAndTest:
    def __init__(self, model, trainloader, valloader=None,
                 optimizer=None, loss=None, device='cpu', print_period=10):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.print_period = print_period

    def train_net(self, epochs):
        self.model.to(self.device)
        criterion = self.loss()
        optimizer = self.optimizer(self.model.parameters(), lr=0.1)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct_train, total_train = 0, 0

            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                if i % self.print_period == self.print_period - 1:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / self.print_period:.3f} '
                          f'Accuracy: {100 * correct_train / total_train:.2f}%')
                    running_loss = 0.0
                    correct_train, total_train = 0, 0

            if self.valloader is not None:
                self.model.eval()
                val_loss = 0.0
                correct_val, total_val = 0, 0

                with torch.no_grad():
                    for data in self.valloader:
                        inputs, labels = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()

                print(f'Validation Loss: {val_loss / len(self.valloader):.3f} '
                      f'Validation Accuracy: {100 * correct_val / total_val:.2f}%')

    def test_net(self):
        if self.testloader is None:
            print("Test loader is not provided.")
            return

        self.model.to(self.device)
        self.model.eval()
        criterion = self.loss()

        test_loss = 0.0
        correct_test, total_test = 0, 0

        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        print(f'Test Loss: {test_loss / len(self.testloader):.3f} '
              f'Test Accuracy: {100 * correct_test / total_test:.2f}%')
