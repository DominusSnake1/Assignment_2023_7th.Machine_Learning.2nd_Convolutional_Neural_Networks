from Classes.TrainAndTest import TrainAndTest
from Classes.MLProjectToDataset import MLProject2Dataset
from Models.SimpleCNN import SimpleCNN
import torch
from torch.utils.data import random_split
import os


def validateDirectory(dir):
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory '{dir}' not found.")

    metadata_path = os.path.join(dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"'metadata.csv' not found in '{dir}'.\nMake sure you have downloaded the dataset (Check \"README.md\")")

    return dir


def main():
    data_dir = validateDirectory('Data/dermoscopy_classification')

    dataset = MLProject2Dataset(data_dir=data_dir)

    seed = 42

    dataset_size = len(custom_dataset.dataset)
    train_size = int(0.6 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SimpleCNN(classes=7)

    trainer = TrainAndTest(model, train_loader, val_loader)

    trainer.train_net(epochs=20)
    trainer.test_net()


if __name__ == "__main__":
    main()
    print("DONE!")