from Classes.MLProjectToDataset import MLProject2Dataset
import torch
from torch.utils.data import random_split


def main():
    ds = MLProject2Dataset(data_dir='/Data')

    seed = 42

    dataset_size = len(ds)
    train_size = int(0.6 * ds)
    val_size = int(0.1 * ds)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        ds, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )


if __name__ == "__main__":
    main()
