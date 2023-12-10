import torch
from tqdm import tqdm


class NetTrainAndTest:
    def __init__(self, model, trainloader, valloader=None, epochs=10, optimizer=None, loss_fn=None, device='cpu',
                 print_period=10):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.print_period = print_period

    def calculate_accuracy(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    def train_and_test(self):
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0
            total_train = 0
            correct_train = 0

            for i, data in enumerate(tqdm(self.trainloader, desc=f'Epoch {epoch + 1}/{self.epochs}', leave=False)):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                if i % self.print_period == self.print_period - 1:
                    avg_loss = running_loss / self.print_period
                    accuracy_train = correct_train / total_train

                    print(f'Iteration {i + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy_train:.4f}')
                    running_loss = 0.0
                    total_train = 0
                    correct_train = 0

            if self.valloader is not None:
                self.model.eval()
                running_val_loss = 0.0
                total_val = 0
                correct_val = 0

                with torch.no_grad():
                    for val_data in tqdm(self.valloader, desc=f'Validation', leave=False):
                        val_inputs, val_labels = val_data
                        val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)

                        val_outputs = self.model(val_inputs)
                        val_loss = self.loss_fn(val_outputs, val_labels)

                        running_val_loss += val_loss.item()

                        _, val_predicted = torch.max(val_outputs, 1)
                        total_val += val_labels.size(0)
                        correct_val += (val_predicted == val_labels).sum().item()

                avg_val_loss = running_val_loss / len(self.valloader)
                accuracy_val = correct_val / total_val

                print(
                    f'Epoch {epoch + 1}/{self.epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy_val:.4f}')

                self.model.train()

        print('Training complete.')
