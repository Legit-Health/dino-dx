import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 100000000000


class DXDataset4DINO(Dataset):
    """ This dataset works with the DXV*_selected_unsampled.csv file of the LegitHealth-DX dataset (any version) """
    def __init__(self, path_data_csv, path_data_images, transform, min_img_size=256, split=None):
        self.path_data_csv = path_data_csv
        self.path_data_images = path_data_images
        self.data_csv = pd.read_csv(self.path_data_csv)
        self.min_img_size = min_img_size
        self.data_csv = self.data_csv.loc[self.data_csv['short_dim'] >= self.min_img_size, :].reset_index(drop=True)
        self.class_to_idx = {c: i for i, c in enumerate(sorted(list(self.data_csv['label'].unique())))}
        self.transform = transform

        # Keeping only part of the dataset (train/val/test)
        # This is only for supervised learning
        self.split = split
        if self.split in ['train', 'val', 'test']:
            if self.split == 'train':
                self.data_csv = self.data_csv.loc[(self.data_csv['is_val'] == 0) & (self.data_csv['is_test'] == 0)]
            elif self.split == "val":
                self.data_csv = self.data_csv.loc[(self.data_csv['is_val'] == 1)]
            else:
                self.data_csv = self.data_csv.loc[(self.data_csv['is_test'] == 1)]
            self.data_csv = self.data_csv.reset_index(drop=True)

        # Creating a 'samples' attribute as in Pytorch's DatasetFolder
        self.samples = []
        for _, row in self.data_csv.iterrows():
            item = os.path.join(self.path_data_images, row['name']), self.class_to_idx[row['label']]
            self.samples.append(item)

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.path_data_images, self.data_csv.loc[idx, 'name'])
        sample = Image.open(img_path)
        sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        target = self.class_to_idx[self.data_csv.loc[idx, 'label']]

        return sample, target
