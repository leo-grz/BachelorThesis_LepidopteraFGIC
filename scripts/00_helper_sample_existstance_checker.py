'''
Script to check if an image exists for a label and a label for an image.
'''

# Imports
import csv
import pandas as pd
import os

# Config

PATH_TO_DATA = 'C:/Users/Leo/Desktop/BA_MothClassification/data/'
PATH_TO_LABELS = PATH_TO_DATA + 'processed/testing_dataset_top20_max50.csv'
PATH_TO_IMAGES = PATH_TO_DATA + 'processed/testing_dataset_top20_max50_images'

csv_file = pd.read_csv(PATH_TO_LABELS)
gbifids = csv_file['gbifID']

missing_files = []
missing_labels = []

for gbifid in gbifids:
    files = [f for f in os.listdir(PATH_TO_IMAGES) if f.startswith(str(gbifid))]
    if not files:
        missing_files.append(gbifid)
        print(f'No image found for label with ID {id}')

for image in os.listdir(PATH_TO_IMAGES):
    gbifid = image.split('_')[0]
    if not int(gbifid) in gbifids.values:
        missing_labels.append(gbifid)
        print(f'No label found for image with ID {gbifid}')
        
print(f'Amount of images: {len(os.listdir(PATH_TO_IMAGES))}')
if len(missing_files) > 0: 
    print(f'There are {len(missing_files)} images missing for labels with IDs: \n{missing_files}')
else:
    print('All images found.')

print(f'Amount of labels: {len(gbifids)}')
if len(missing_labels) > 0:
    print(f'There are {len(missing_labels)} labels missing for images with IDs: \n{missing_labels}')
else:
    print('All labels found.')