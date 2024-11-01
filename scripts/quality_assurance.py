
# This script takes a tensor file (244x244x3) and assigns them to either blacklist or manualcheck
# it does not look at any images contained in manualcheck, blacklist or whitelist

# should a qa_checked_and_approved.txt file be used?

# Imports
import sys 
import os 
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from moth_dataset import MothDataset

# Config

PATH_TO_PROCESSED = 'C:/Users/Leo/Desktop/BA_MothClassification/data/processed/'
PATH_TO_LABELS = PATH_TO_PROCESSED + 'testing_dataset_top20x50.csv'
PATH_TO_IMAGES = PATH_TO_PROCESSED + 'testing_dataset_top20x50_images'


# prepare rows not containing BLACK / CHECK to be inspected by brightness check and objective size estimation
csv_file = pd.read_csv(PATH_TO_LABELS)
csv_file['status'] = csv_file['status'].astype('str')
csv_file_filtered = csv_file[~csv_file['status'].isin(['CHECK', 'BLACK'])]
csv_file_for_loader = csv_file_filtered[['gbifID', 'scientificName']]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),          # Convert to tensor
])

full_dataset = MothDataset(csv_file=csv_file_for_loader, root_dir=PATH_TO_IMAGES, transform=transform)

# only process files who's filenames are not already in whitelist, manualcheck or blacklist

dataloader = DataLoader(full_dataset, batch_size=100, shuffle=False)

blacklist = []
checklist = []

# Brightness check

def calculate_darkness(tensor):
    grayscale_tensor = tensor.mean(dim=0) # convert to grayscale
    average_intensity = grayscale_tensor.mean().item() # calculate average pixel intensity

    darkness_abs_threshold = 0.05
    darkness_threshold = 0.15
    is_dark = average_intensity < darkness_threshold
    is_black = average_intensity < darkness_abs_threshold

    return average_intensity, is_dark, is_black


i = 0
for images, labels, gbifids, img_names in dataloader:
    i +=1
    print(f'Bach [{i}/{len(dataloader)}]')
    
    for image, label, gbifid, img_name in zip(images, labels, gbifids, img_names):

        average_intensity, is_dark, is_black = calculate_darkness(image)
        if is_black: # if image is without a doubt too dark, automatically add it to blacklist
            blacklist.append(int(gbifid))
            print(f'Average Intensity: {average_intensity} | Adding {img_name} to blacklist.')

        elif is_dark: # add image to manual check list
            checklist.append(int(gbifid))
            print(f'Average Intensity: {average_intensity} | Adding {img_name} to checklist.')

# Size of object check


for gbifid in checklist:
    csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'CHECK'

for gbifid in blacklist:
    csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'BLACK'

csv_file.to_csv(PATH_TO_LABELS, index=False)
