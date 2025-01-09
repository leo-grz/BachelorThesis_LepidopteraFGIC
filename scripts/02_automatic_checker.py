'''
This script takes an image as tensor (244x244x3) and assigns them to either blacklist or manualcheck
it does not look at any images contained in manualcheck, blacklist or whitelist. The outcomes of the check
are written to the csv file in the status column.
It's the script to be executed before moving on the manual_checker.py
'''
# Imports
import sys 
import os 
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from lepidoptera_dataset import LepidopteraDataset
import torch
import logging
import gc
# Config

PATH_TO_DATA = '/home/lgierz/BA_MothClassification/data/'
PATH_TO_LABELS = PATH_TO_DATA + 'processed/dataset_top589_max3000.csv'
PATH_TO_IMAGES = '/mnt/data/lgierz/moth_dataset_top589_max3000'
PATH_TO_LOGFILE = PATH_TO_DATA + 'status/automatic_checker.log'

DARKNESS_BLACKLIST_THRESHOLD = 0.01
DARKNESS_CHECKLIST_THRESHOLD = 0.06

STATES_TO_IGNORE= ['CHECK', 'BLACK', 'MISSING', 'SEEN', 'ERROR']


logging.basicConfig(
    filename=PATH_TO_LOGFILE,
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
)

# prepare rows not containing BLACK / CHECK to be inspected by brightness check and objective size estimation
csv_file = pd.read_csv(PATH_TO_LABELS)
csv_file['status'] = csv_file['status'].astype('str') # to ensure status (CHECK, WHITE, BLACK) is of type string

csv_file_filtered = csv_file[~csv_file['status'].isin(STATES_TO_IGNORE)] # selects all samples which's status has not been set to CHECK or BLACK or MISSING
csv_file_filtered.reset_index(drop=True, inplace=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),          # Convert to tensor
])

full_dataset = LepidopteraDataset(csv_file=csv_file_filtered, root_dir=PATH_TO_IMAGES, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'[ok] Device chosen: {device}')
logging.info(f'[INIT] Device chosen: {device}')
dataloader = DataLoader(full_dataset, batch_size=1000, shuffle=False, num_workers=24, prefetch_factor=2)#, pin_memory=True)

print('[ok] Initialized Dataloader.')
logging.info(f'[INIT] Initialized Dataloader')

def calculate_darkness(tensor):
    ''' 
    Calculate the darkness of an image tensor. 
    
    The function converts an RGB tensor to grayscale, computes the average pixel intensity, 
    and determines returns values based on which the further computation of the sample is determined. 
    
    Parameters: 
    tensor (torch.Tensor): A tensor of shape (3, H, W) representing an RGB image. 
    
    Returns: 
    tuple: 
        - average_intensity (float): The average intensity of the grayscale image. 
        - is_dark (bool): True if the average intensity is below the darkness threshold. 
        - is_black (bool): True if the average intensity is below the absolute darkness threshold. 
    '''
    grayscale_tensor = tensor.mean(dim=0) # convert to grayscale
    average_intensity = grayscale_tensor.mean().item() # calculate average pixel intensity

    is_dark = average_intensity < DARKNESS_CHECKLIST_THRESHOLD
    is_black = average_intensity < DARKNESS_BLACKLIST_THRESHOLD

    return average_intensity, is_dark, is_black

checklist, blacklist, seenlist, errorlist = [], [], [], []
seen_count = 0
error_count = 0
black_count = 0
check_count = 0

for batch, (images, labels, gbifids, img_names) in enumerate(dataloader, start=1):
    print(f'Batch [{batch}/{len(dataloader)}] with {len(gbifids)} samples')
    logging.info(f'[BATCH][{batch}/{len(dataloader)}] with {len(gbifids)} samples')
    images = images.to(device)
    
    
    for image, label, gbifid, img_name in zip(images, labels, gbifids, img_names):

        if int(label) < 0: # if image label is negative, it indicates an error
            errorlist.append(int(gbifid))
            real_label = (int(label)-1) * -1
            logging.error(f'[SAMPLE][ERROR] Adding {img_name} to errorlist. (Label: {real_label} | gbifID: {gbifid})')
            continue
        
        average_intensity, is_dark, is_black = calculate_darkness(image)

        if is_black: # if image is without a doubt too dark, automatically add it to blacklist
            blacklist.append(int(gbifid))
            print(f' Average Intensity: {average_intensity} | Adding {img_name} to blacklist.')
            logging.info(f'[SAMPLE][BLACK] Adding {img_name} to blacklist. (Label: {int(label)} | gbifID: {gbifid})')
        elif is_dark: # add image to manual check list
            checklist.append(int(gbifid))
            print(f'Average Intensity: {average_intensity} | Adding {img_name} to checklist.')
            logging.info(f'[SAMPLE][CHECK] Adding {img_name} to checklist.')
        else:
            seenlist.append(int(gbifid))
    
    if batch % 50 == 0:
        gc.collect() 
        torch.cuda.empty_cache()
        

check_count += len(checklist)
black_count += len(blacklist)
seen_count += len(seenlist)
error_count += len(errorlist)

        
for gbifid in checklist: # write CHECK status to csv file, these samples will manually be checked in manual_check.py
    csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'CHECK'

for gbifid in blacklist: # write BLACK status to csv file
    csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'BLACK'

for gbifid in seenlist: # write SEEN status to csv file
    csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'SEEN'

for gbifid in errorlist: # write ERROR status to csv file
    csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'ERROR'

logging.info(f'[FINAL REPORT][BLACK] Samples added to blacklist: {black_count}')
logging.info(f'[FINAL REPORT][CHECK] Samples added to checklist: {check_count}')
logging.info(f'[FINAL REPORT][SEEN] Samples added to seenlist: {seen_count}')
logging.info(f'[FINAL REPORT][ERROR] Samples added to errorlist: {error_count}')

csv_file.to_csv(PATH_TO_LABELS, index=False) # save updated statuses to csv file
logging.info(f'[FINAL REPORT] Amount of statuses updated in csv: {black_count + error_count + seen_count + check_count} | Exiting...')

print('[ok] Done. Exiting...')

