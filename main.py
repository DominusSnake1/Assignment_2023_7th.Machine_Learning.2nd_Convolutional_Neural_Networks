from Classes.MLProjectToDataset import MLProject2Dataset
from Classes.NetTrainAndTest import NetTrainAndTest
from torch.utils.data import random_split, DataLoader
from Models.SimpleCNN import SimpleCNN
from torchvision import transforms
import torch


def transformation(m, n):
    transform = transforms.Compose([
        transforms.Resize((m, n), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform


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

    model = SimpleCNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = NetTrainAndTest(model=model, trainloader=train_loader,
                              epochs=20, optimizer=optimizer, loss_fn=criterion)

    trainer.train_net()


if __name__ == "__main__":
    main()
