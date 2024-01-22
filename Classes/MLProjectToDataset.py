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
        self.metadata = pd.read_csv(metadata_path)

        self.metadata['dx'] = pd.Categorical(self.metadata['dx']).codes

        self.data_info = pd.merge(self.data_info, self.metadata[['image_id', 'dx']], on='image_id')

        self.demographic_features = self.process_demographic_features()

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path = self.data_info.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")

        label = int(self.data_info.iloc[idx, 2])

        if self.transform is not None:
            image = self.transform(image)

        # demographic_features = torch.from_numpy(self.demographic_features[idx]).float()
        # return image, demographic_features, label
        return image, label

    def process_demographic_features(self):
        categorical_features = pd.get_dummies(self.metadata[['sex', 'localization']],
                                              columns=['sex', 'localization'])

        demographic_features = categorical_features.to_numpy()

        demographic_features[:, -1] = self.metadata['age'] / 100

        return demographic_features


def split_dataset(transformations, data_directory='Data/'):
    dataset = MLProject2Dataset(
        data_dir=data_directory,
        transform=transformations
    )

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
