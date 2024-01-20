import numpy as np
import torch


class NetTrainAndTest:
    def __init__(self, model, trainloader, testloader, valloader=None, epochs=10, optimizer=None, loss_fn=None, print_period=10):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = 'cpu'
        self.print_period = print_period

    def accuracy(self, output, target):
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).sum().item()
        total = target.size(0)
        return correct / total

    def train_net(self):
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0
            total_accuracy = 0.0

            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_accuracy += self.accuracy(outputs, labels)

                if i % self.print_period == self.print_period - 1:
                    avg_loss = running_loss / self.print_period
                    avg_accuracy = total_accuracy / self.print_period
                    print(f'[Epoch {epoch + 1}, Iteration {i + 1}] Average Loss: {avg_loss:.3f}, Average Accuracy: {avg_accuracy:.3f}')
                    running_loss = 0.0
                    total_accuracy = 0.0

            # Validate the model after each epoch on the validation set
            if self.valloader is not None:
                self.model.eval()
                val_loss = 0.0
                val_accuracy = 0.0

                with torch.no_grad():
                    for data in self.valloader:
                        inputs, labels = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        outputs = self.model(inputs)
                        val_loss += self.loss_fn(outputs, labels).item()
                        val_accuracy += self.accuracy(outputs, labels)

                avg_val_loss = val_loss / len(self.valloader)
                avg_val_accuracy = val_accuracy / len(self.valloader)
                print(f'\nValidation Loss: {avg_val_loss:.3f}, Validation Accuracy: {avg_val_accuracy:.3f}\n')

                self.model.train()

        print('Training finished')

    def test_net(self):
        self.model.to(self.device)
        self.model.eval()

        test_loss = 0.0
        test_accuracy = 0.0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                test_loss += self.loss_fn(outputs, labels).item()
                test_accuracy += self.accuracy(outputs, labels)

                predictions.append(outputs.argmax(dim=1).cpu().numpy())
                true_labels.append(labels.cpu().numpy())

        avg_test_loss = test_loss / len(self.testloader)
        avg_test_accuracy = test_accuracy / len(self.testloader)

        print(f'Test Loss: {avg_test_loss:.3f}, Test Accuracy: {avg_test_accuracy:.3f}')

        predictions = np.concatenate(predictions)
        true_labels = np.concatenate(true_labels)

        self.model.train()

        return predictions, true_labels
