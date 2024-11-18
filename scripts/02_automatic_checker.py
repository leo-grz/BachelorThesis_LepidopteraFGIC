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

# Config

PATH_TO_DATA = 'C:/Users/Leo/Desktop/BA_MothClassification/data/'
PATH_TO_LABELS = PATH_TO_DATA + 'processed/testing_dataset_top20_max50.csv'
PATH_TO_IMAGES = PATH_TO_DATA + 'processed/testing_dataset_top20_max50_images'

DARKNESS_BLACKLIST_THRESHOLD = 0.05
DARKNESS_CHECKLIST_THRESHOLD = 0.1

STATES_TO_IGNORE= ['CHECK', 'BLACK', 'MISSING']


# prepare rows not containing BLACK / CHECK to be inspected by brightness check and objective size estimation
csv_file = pd.read_csv(PATH_TO_LABELS)
csv_file['status'] = csv_file['status'].astype('str') # to ensure status (CHECK, WHITE, BLACK) is of type string

if 'NEW' in csv_file['status'].values:
    csv_file_filtered = csv_file[csv_file['status'] == 'NEW'] # select all samples with status NEW if they exist
    mode_new = True
else:
    csv_file_filtered = csv_file[~csv_file['status'].isin(STATES_TO_IGNORE)] # selects all samples which's status has not been set to CHECK or BLACK or MISSING
    mode_new = False

csv_file_filtered.reset_index(drop=True, inplace=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),          # Convert to tensor
])

full_dataset = LepidopteraDataset(csv_file=csv_file_filtered, root_dir=PATH_TO_IMAGES, transform=transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'[ok] Device chosen: {device}')
dataloader = DataLoader(full_dataset, batch_size=1000, shuffle=False)
print('[ok] Initialized Dataloader.')

blacklist = []
checklist = []
ignorelist = []

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


for batch, (images, labels, gbifids, img_names) in enumerate(dataloader, start=1):
    print(f'Bach [{batch}/{len(dataloader)}]')
    
    images = images.to(device)
    
    for image, label, gbifid, img_name in zip(images, labels, gbifids, img_names):
        if mode_new:
            print(f'[MODE_NEW] Processing Sample with ID {gbifid} and Filename {img_name}')

        average_intensity, is_dark, is_black = calculate_darkness(image)

        if is_black: # if image is without a doubt too dark, automatically add it to blacklist
            blacklist.append(int(gbifid))
            print(f'Average Intensity: {average_intensity} | Adding {img_name} to blacklist.')

        elif is_dark: # add image to manual check list
            checklist.append(int(gbifid))
            print(f'Average Intensity: {average_intensity} | Adding {img_name} to checklist.')
        else:
            ignorelist.append(int(gbifid))


if input('Press 1 to update the csv file.') == '1':
    print('[ok] Updating csv file.')

    for gbifid in checklist: # write CHECK status to csv file, these samples will manually be checked in manual_check.py
        csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'CHECK'

    for gbifid in blacklist: # write BLACK status to csv file
        csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'BLACK'

    for gbifid in ignorelist: # write BLACK status to csv file
        csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'IGNORED'

    csv_file.to_csv(PATH_TO_LABELS, index=False) # save updated statuses to csv file

    print('[ok] Done. Exiting...')
else:
    print('[!!] Not updating csv file. Exiting...')
