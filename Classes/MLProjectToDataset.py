import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import glob


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

    def __getitem__(self, index):
        img_path = self.data_info.iloc[index, 0]
        image = Image.open(img_path).convert('RGB')

        label = int(self.data_info.iloc[index, 2])

        if self.transform is not None:
            image = self.transform(image)

        return image, label
