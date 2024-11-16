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
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from moth_dataset import MothDataset

# Config

PATH_TO_DATA = 'C:/Users/Leo/Desktop/BA_MothClassification/data/'
PATH_TO_LABELS = PATH_TO_DATA + 'processed/testing_dataset_top20_max50.csv'
PATH_TO_IMAGES = PATH_TO_DATA + 'processed/testing_dataset_top20_max50_images'

DARKNESS_BLACKLIST_THRESHOLD = 0.05
DARKNESS_CHECKLIST_THRESHOLD = 0.15
CONTOUR_CHECKLIST_THRESHOLD = 50

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

csv_file_for_ds = csv_file_filtered[['gbifID', 'scientificName']] # to pass only relevant fields to MothDataset Class

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),          # Convert to tensor
])

full_dataset = MothDataset(csv_file=csv_file_for_ds, root_dir=PATH_TO_IMAGES, transform=transform)
dataloader = DataLoader(full_dataset, batch_size=100, shuffle=False)

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


def estimate_object_size(tensor):
    """
    Calculate the size of the biggest object in the image.

    This function processes an image tensor, converts it to grayscale, applies edge detection,
    and finds the largest contour to estimate the object's size in pixels.

    Parameters: 
    tensor (torch.Tensor): A tensor of shape (3, H, W) representing an RGB image.

    Returns:
    tuple:
        - area (int): Size of the biggest object in pixels (area of the largest contour).
        - is_small (bool): True if the area is below the size threshold. 
    """
    array = tensor.numpy().transpose((1, 2, 0))
    image = np.clip(array * 255, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert the image to grayscale
    edges = cv2.Canny(gray, 100, 200)  # Apply Canny edge detector
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

    if contours:  # Calculate the area of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        is_small = area < CONTOUR_CHECKLIST_THRESHOLD
        return area, is_small
    else:
        return 0, False

def is_contour_at_edge(contour, img_shape):
    for point in contour:
        if point[0][0] == 0 or point[0][1] == 0 or point[0][0] == img_shape[1] - 1 or point[0][1] == img_shape[0] - 1:
            return True
    return False

def filter_image(tensor, contour_threshold=100):
    array = tensor.numpy().transpose((1, 2, 0))
    image = np.clip(array * 255, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert the image to grayscale
    edges = cv2.Canny(gray, 50, 100)  # Apply Canny edge detector with adjusted thresholds
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    valid_contours = [contour for contour in contours if not is_contour_at_edge(contour, image.shape)]
    valid_contours = [contour for contour in contours if 500 < cv2.contourArea(contour) < 35000]

    # for contour in contours:
    #     print(int(cv2.contourArea(contour)))
    #     print(500 < int(cv2.contourArea(contour)) < 40000)

    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Filter based on the number of valid contours and the area of the largest valid contour
        if len(valid_contours) > 5:
            print(f"Filtered due to multiple butterflies. Contours: {len(valid_contours)}")
            return area, True
        elif area < contour_threshold:
            print(f"Filtered due to weak contour. Area: {area}")
            return area, True
        else:
            return area, False
    else:
        print("Filtered due to no valid contours.")
        return 0, True

i = 0
for images, labels, gbifids, img_names in dataloader:
    i +=1
    print(f'Bach [{i}/{len(dataloader)}]')
    
    for image, label, gbifid, img_name in zip(images, labels, gbifids, img_names):
        if mode_new:
            print(f'[MODE_NEW] Processing Sample with ID {gbifid} and Filename {img_name}')

        average_intensity, is_dark, is_black = calculate_darkness(image)
        object_size, is_small = estimate_object_size(image)
        #object_size, is_small = filter_image(image)

        if is_black: # if image is without a doubt too dark, automatically add it to blacklist
            blacklist.append(int(gbifid))
            print(f'Average Intensity: {average_intensity} | Addi
                  ng {img_name} to blacklist.')

        elif is_dark or is_small: # add image to manual check list
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
