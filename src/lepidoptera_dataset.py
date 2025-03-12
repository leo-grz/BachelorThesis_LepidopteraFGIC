import torch
from torch.utils.data import Dataset

from PIL import Image, UnidentifiedImageError
import os
import logging

PATH_TO_DATA = '/home/lgierz/BA_MothClassification/data/'
PATH_TO_LOGFILE = PATH_TO_DATA + 'status/lepidoptera_dataset.log'

logging.basicConfig(
    filename=PATH_TO_LOGFILE,
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
)

def get_labelencoding(df):
    species_list = sorted(df['scientificName'].unique())
    species_dict = {species: idx for idx, species in enumerate(species_list)}
    return species_dict


class LepidopteraDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        #self.data_frame = pd.read_csv(csv_file)[['gbifID', 'scientificName']]
        self.data_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform
        #self.label_encoder = LabelEncoder()
        self.mapping = get_labelencoding(csv_file)
        #self.data_frame['scientificName_encoded'] = self.label_encoder.fit_transform(self.data_frame['scientificName'])

        # Build an in-memory index mapping gbifID to file paths
        self.index = {}
        for file_name in os.listdir(self.root_dir):
            if file_name.endswith('.jpg'):
                gbif_id = file_name.split('_')[0]
                self.index[gbif_id] = os.path.join(self.root_dir, file_name)
        print("[LEPIDOPTERA_DATASET] In-memory index built.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        #_st = pc()
        gbifid = self.data_frame.loc[idx, 'gbifID']
        img_path = self.index[str(gbifid)]  # Select the first matching file from the index
        label_name = self.data_frame.loc[idx, 'scientificName']
        label = self.mapping[label_name]

        try: 
            with Image.open(img_path) as img:  # use OpenCV for faster operations
                image = img.convert("RGB")
            image_tensor = self.transform(image)
            label_tensor = torch.tensor(label, dtype=torch.long)

        except (UnidentifiedImageError, OSError) as e:
            image_tensor = torch.rand((3,224,224)) # random tensor to not confuse data loader
            label = (label + 1) * -1 # to keep the label while addressing error (+1 bc -0 doesnt exist)
            label_tensor = torch.tensor(label, dtype=torch.long)
            #print(f'[LEPIDOPTERA_DATASET][ERROR] {e}')
            logging.error(f'[LEPIDOPTERA_DATASET] Image with gbifID {gbifid} couldn\'t be processed. Error: {e}')
        #_dur = pc() - _st
        #if _dur > 0.5: logging.error(f'[LEPIDOPTERA_DATASET] IMAGE WITH ID {gbifid} TOOK {_dur} SECONDS')
        return image_tensor, label_tensor, gbifid, img_path.split('/')[-1]  # Returning the image, label, ID and img_name

    def decode_label(self, encoded_label):
        '''Helper Function to display the type of moth, reversing label encoder.'''
        return self.label_encoder.inverse_transform([encoded_label])[0]


