import os
import pandas as pd
from torchvision.io import read_image
from .base_dataset import BaseDataset


class TrainDataset(BaseDataset):
    def __init__(self, root_dir, csv_path, transform=None):
        super().__init__(transform)

        self.root_dir = root_dir
        df = pd.read_csv(csv_path)

        self.filenames = df["Id"].values
        self.labels = df["Category"].values

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = int(self.labels[idx])

        img_path = os.path.join(self.root_dir, fname)
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

class TestDataset(BaseDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(transform)

        self.root_dir = root_dir
        self.filenames = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.root_dir, fname)

        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, fname

