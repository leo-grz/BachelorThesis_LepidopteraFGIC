from torchvision import transforms
import time
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder

from PIL import Image
import os

class LepidopteraDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        #self.data_frame = pd.read_csv(csv_file)[['gbifID', 'scientificName']]
        self.data_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.data_frame['scientificName_encoded'] = self.label_encoder.fit_transform(self.data_frame['scientificName'])

        # Build an in-memory index mapping gbifID to file paths
        self.index = {}
        for file_name in os.listdir(self.root_dir):
            if file_name.endswith('.jpg'):
                gbif_id = file_name.split('_')[0]
                self.index[gbif_id] = os.path.join(self.root_dir, file_name)
        print("[ok] In-memory index built.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        gbif_id = self.data_frame.loc[idx, 'gbifID']
        img_path = self.index[str(gbif_id)]  # Select the first matching file from the index
        image = Image.open(img_path).convert("RGB")
        label = self.data_frame.loc[idx, 'scientificName_encoded']
        gbifid = self.data_frame.loc[idx, 'gbifID']
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor, gbifid, img_path.split('/')[-1]  # Returning the image, label, ID and img_name

    def decode_label(self, encoded_label):
        '''Helper Function to display the type of moth, reversing label encoder.'''
        return self.label_encoder.inverse_transform([encoded_label])[0]
