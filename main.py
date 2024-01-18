from Classes.MLProjectToDataset import MLProject2Dataset
from torch.utils.data import random_split, DataLoader
from Classes.NetTrainAndTest import NetTrainAndTest
from Models.ComplexCNN import ComplexCNN
from Models.SimpleCNN import SimpleCNN
from Other.Utils import transformation
from torchsummary import summary
import torch


def chooseModel(model, train_loader, test_loader, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    trainer = NetTrainAndTest(model=model, trainloader=train_loader,
                              testloader=test_loader, epochs=20,
                              optimizer=optimizer, loss_fn=criterion)

    summary(model, (3, 50, 62))

    trainer.train_net()
    trainer.test_net()


def main():
    dataset = MLProject2Dataset(data_dir='Data/', transform=transformation(50, 62))

    train_percentage = 0.6
    val_percentage = 0.1
    test_percentage = 0.3

    num_data = len(dataset)
    num_train = int(train_percentage * num_data)
    num_val = int(val_percentage * num_data)
    num_test = num_data - num_train - num_val

    train_set, val_set, test_set = random_split(dataset, [num_train, num_val, num_test],
                                                generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # Simple CNN
    chooseModel(model=SimpleCNN(), train_loader=train_loader, test_loader=test_loader, learning_rate=0.1)

    # Complex CNN
    chooseModel(model=ComplexCNN(), train_loader=train_loader, test_loader=test_loader, learning_rate=1e-3)


if __name__ == "__main__":
    main()
