'''
This script extracts features from images using a specified model (ResNet or DINO).
It processes images, extracts features, and saves them to a file. It can also validate
the extracted features against previously calculated features.

The script should be executed after preparing the dataset and before any further analysis.
'''

# Imports
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torchvision import transforms
import torchvision.models as models

import logging
import numpy as np
import pandas as pd
import os
import sys
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from lepidoptera_dataset import LepidopteraDataset
from utils_helpers import check_folder_exists, load_features, save_features

# Configuration

PATH_TO_DATA = '/home/lgierz/BA_MothClassification/data/'
PATH_TO_DATASETS = PATH_TO_DATA + 'processed/cv_datasets/'
PATH_TO_IMAGES = '/mnt/data/lgierz/moth_dataset_top589_max3000/'
PATH_TO_FEATURES = PATH_TO_DATA + 'processed/features/'

# check existance of folders
for folder in [PATH_TO_DATA, PATH_TO_DATASETS, PATH_TO_IMAGES, PATH_TO_FEATURES]:
    check_folder_exists(folder, min_fileamount=0)

PATH_TO_LOGFILE = PATH_TO_DATA + 'status/dino_feature_extraction.log'
DATASET = PATH_TO_DATA + 'processed/features/dataset_top589_max3000_fex_statusupdate.csv'
MODEL_NAME = 'dino'


# set mode to validation
VALIDATE_FEATURES = False
if VALIDATE_FEATURES:
    VALIDATE_FEATURES_AMOUNT = 10000
    FEATURES_TO_VALIDATE = PATH_TO_FEATURES + 'ResNet50_Features_Full_CorrectlyLabeled.npz'
    VAL_F, VAL_L, VAL_G = load_features(FEATURES_TO_VALIDATE)


# Configure logging
logging.basicConfig(
    filename='/home/lgierz/BA_MothClassification/data/status/feature_extraction.log',
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
)

console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')) 
logger = logging.getLogger() 
logger.addHandler(console_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f'[INIT] Device chosen: {device}')

csv_file = pd.read_csv(DATASET)
csv_file['status'] = csv_file['status'].astype('str') # to ensure status (CHECK, WHITE, BLACK) is of type string

ignore_statuses = ['CHECK', 'BLACK', 'MISSING', 'ERROR', 'DFEX1'] # these statuses are ignored in dataset
csv_file_filtered = csv_file[~csv_file['status'].isin(ignore_statuses)] # selects all samples which's status has not been set to CHECK or BLACK


# randomly sample samples to check for match between previously calculated features
if VALIDATE_FEATURES: csv_file_filtered = csv_file_filtered.sample(n=VALIDATE_FEATURES_AMOUNT, random_state=42)

csv_file_filtered.reset_index(drop=True, inplace=True)

#proceed = input(f"Do you want to proceed? {len(csv_file_filtered['gbifID'])} samples are going to be processed. y/n ")
#if not proceed == 'y':
#    print('Abort...')
#    sys.exit(0)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),          # Convert to tensor
])

full_dataset = LepidopteraDataset(csv_file=csv_file_filtered, root_dir=PATH_TO_IMAGES, transform=transform)
logging.info(f'Created dataset with {len(full_dataset)} samples.')

full_loader = DataLoader(full_dataset, batch_size=500, shuffle=False, num_workers=20, prefetch_factor=2, pin_memory=True)
logging.info(f'Created loader with {len(full_loader)} batches.')



if MODEL_NAME == 'resnet':
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()    
elif MODEL_NAME == 'dino':
    #model = models.vit_b_16(weights=models.ViT_B_16_Weights.DINO)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

    print(model)
    model.head = nn.Identity()
else:
    logging.critical(f'No model called "{MODEL_NAME}"! Exiting...')
    sys.exit(1)

gc.collect() 
torch.cuda.empty_cache()


model = model.to(device)
model.eval()

features = []
labels = []
gbifid_list = []

with torch.no_grad():
    print('Starting feature extraction process...')
    for batch, (images, lbls, gbifids, _ ) in enumerate(full_loader, start=1):

        logging.info(f'[FEATURE EXTRACTION] Batch [{batch}/{len(full_loader)}] with {len(images)} samples')
        valid_indices = []

        for idx, lbl in enumerate(lbls): # sorting out images that could not be read
            if int(lbl) >= 0: # valid label if 0 or above, if negative, there has been an error with the sample
                valid_indices.append(idx)
            else:
                logging.error(f'[FEATURE EXTRACTION] Image with gbifID {gbifids[idx]} couldn\'t be read.')

        images = images[valid_indices]
        lbls = lbls[valid_indices]
        gbifids = gbifids[valid_indices]

        images = images.to(device)
        outputs = model(images).cpu().numpy()

        features.append(outputs)
        labels.append(lbls.numpy())
        gbifid_list.append(gbifids.numpy())


        if (batch % 40 == 0 or batch == len(full_loader)) and not VALIDATE_FEATURES:
            # if not in validate_features mode and every 40th batch and at the last processed batch in dataloader:
            features = np.concatenate(features)
            labels = np.concatenate(labels)
            gbifid_list = np.concatenate(gbifid_list)

            PATH_TO_FEATURES_FILE = PATH_TO_FEATURES + 'features_DINOv2_top589_max3000_File3.npz'
            save_features(features, labels, gbifid_list, PATH_TO_FEATURES_FILE)
            logging.info(f'[REPORT] Saved features at batch {batch}')

            for gbifid in list(gbifid_list):
                csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'DFEX1' # indicates Feature Extraction second round (FEX2) -> DINOv2

            csv_file.to_csv(DATASET, index=False) # save updated statuses to csv file

            features = []
            labels = []
            gbifid_list = []

            gc.collect() 
            torch.cuda.empty_cache()


        # to check if the features are correct and ensure no mistake has been made
        elif VALIDATE_FEATURES:
            
            features = np.concatenate(features)
            gbifid_list = np.concatenate(gbifid_list)
            labels = np.concatenate(labels)

            match_counter = 0
            label_mismatch_counter = 0
            embedding_mismatch_counter = 0
            # Create a dictionary to map gbifid to its features 
            gbifid_to_features = {gbifid: (VAL_F[i], VAL_L[i]) for i, gbifid in enumerate(VAL_G)} 
            # Compare the features for each gbifid 
            for i, gbifid in enumerate(gbifid_list): 

                tolerance = 1e-3
                if not np.allclose(features[i], gbifid_to_features[gbifid][0], atol=tolerance): # due to rounding errors roughly 0.5% of features are slighly uneven -> tolerance
                
                    logging.error(f"[FEATURE VALIDATION] Mismatching FEATURE EMBEDDING found for gbifid {gbifid}") 
                    # print(f'Loaded: {gbifid_to_features[gbifid][0]}')
                    # print(f'Calculated: {features[i]}')
                    embedding_mismatch_counter += 1
                elif labels[i] != gbifid_to_features[gbifid][1]: # check if labels match
                    logging.error(f"[FEATURE VALIDATION] Mismatching LABEL found for gbifid {gbifid}") 
                    # print(f'Loaded: {gbifid_to_features[gbifid][1]}')
                    # print(f'Calculated: {labels[i]}')
                    label_mismatch_counter += 1

                else: 
                    match_counter += 1

            logging.info(f'[FEATURE VALIDATION] Comparison for batch [{batch}/{len(full_loader)}]: {match_counter} matches and {label_mismatch_counter + embedding_mismatch_counter} mismatches [{label_mismatch_counter}L | {embedding_mismatch_counter}F]')
            features, gbifid_list, labels = [], [], []






