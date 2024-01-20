from torch.utils.data import random_split, DataLoader, Dataset
from PIL import Image
import pandas as pd
import torch
import glob
import os


class MLProject2Dataset(Dataset):
    def __init__(self, data_dir, metadata_fname='metadata.csv', transform=None):
        self.data_dir = data_dir
        self.transform = transform

        image_paths = glob.glob(os.path.join(data_dir, '*.jpg'))
        image_ids = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]

        self.data_info = pd.DataFrame({
            'image_id': image_ids,
            'path': image_paths
        })

        metadata_path = os.path.join(data_dir, metadata_fname)
        metadata = pd.read_csv(metadata_path)

        metadata['dx'] = pd.Categorical(metadata['dx']).codes

        self.data_info = pd.merge(self.data_info, metadata[['image_id', 'dx']], on='image_id')

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path = self.data_info.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")

        label = int(self.data_info.iloc[idx, 2])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def split_dataset(dataset):
    num_data = len(dataset)
    num_train = int(0.6 * num_data)
    num_val = int(0.1 * num_data)
    num_test = num_data - num_train - num_val

    train_set, val_set, test_set = random_split(
        dataset=dataset,
        lengths=[num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=64,
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=64,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
