import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from Classes.Transformations import Transformations


class MLProject2Dataset(Dataset):
    def __init__(self, data_dir, metadata_fname='metadata.csv'):
        self.data_dir = data_dir
        self.transform = Transformations(m=128, n=128)

        # Read metadata
        metadata_path = os.path.join(data_dir, metadata_fname)
        metadata = pd.read_csv(metadata_path)
        metadata['dx'] = pd.Categorical(metadata['dx']).codes

        # Create DataFrame with image paths and labels
        image_paths = [os.path.join(data_dir, f) for f in metadata['image_id']]
        labels = metadata['dx']
        self.data = pd.DataFrame({'path': image_paths, 'label': labels})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long)

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img, label
