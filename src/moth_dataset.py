from torchvision import transforms

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder

import glob # for wild card when image_name selection
from PIL import Image
import os
import pandas as pd

class MothDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        #self.data_frame = pd.read_csv(csv_file)[['gbifID', 'scientificName']]
        self.data_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.data_frame['scientificName_encoded'] = self.label_encoder.fit_transform(self.data_frame['scientificName'])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = glob.glob(os.path.join(self.root_dir, f"{self.data_frame.iloc[idx, 0]}_*.jpg"))[0] # select the first tile matching the pattern
        image = Image.open(img_name).convert("RGB")
        label = self.data_frame.iloc[idx, 2]  # the idx 2 column contains the scientificName_encoded
        label_name = self.data_frame.iloc[idx, 1]
        gbifid = self.data_frame.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor, gbifid, img_name.split('\\')[-1]  # Returning the image, label, and ID

    def decode_label(self, encoded_label):
        '''Helper Function to display the type of moth, reversing label encoder.'''
        return self.label_encoder.inverse_transform([encoded_label])[0]
