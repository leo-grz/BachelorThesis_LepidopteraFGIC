'''
Script to check if an image exists for a label and a label for an image.
'''

# Imports
import pandas as pd
import os

# Config

PATH_TO_DATA = '/home/lgierz/BA_MothClassification/data/processed/'
PATH_TO_LABELS = PATH_TO_DATA + 'dataset_top589_max3000.csv'
PATH_TO_IMAGES = '/mnt/data/lgierz/moth_dataset_top589_max3000/'

print(f'[ok] Readng in csv file: {PATH_TO_LABELS}')
csv_file = pd.read_csv(PATH_TO_LABELS)
csv_file['status'] = csv_file['status'].astype('str')

print(f'[ok] Creating label gbifid list.')
label_gbifids = csv_file['gbifID'].values
label_gbifids = [int(gbifid) for gbifid in label_gbifids]

print(f'[ok] Creating image gbifid list.')
listdir = os.listdir(PATH_TO_IMAGES)
image_gbifids = [int(filename.split('_')[0]) for filename in listdir]

print(f'[ok] Converting gbif lists to sets.')
# Convert lists to sets 
label_gbifids_set = set(label_gbifids) 
image_gbifids_set = set(image_gbifids) 

print(f'[ok] Comparing sets.\n')
# Find values unique to each list 
unique_label_gbifids = label_gbifids_set - image_gbifids_set 
unique_image_gbifids = image_gbifids_set - label_gbifids_set 

# Convert sets back to lists (if needed) 
unique_label_gbifids = list(unique_label_gbifids) 
unique_image_gbifids = list(unique_image_gbifids) 

# Print the results 
print(f"Unique Label gbifIDs: {len(unique_label_gbifids)}") 
print(f"Unique Image gbifIDs: {len(unique_image_gbifids)}")


# Update statuses
if input('Press 1 to change status of missing images for labels to "MISSING"') == '1':
    print('[ok] Updating statuses.')
    for gbifid in unique_label_gbifids: # write MISSING status to csv file
        csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'MISSING'
    csv_file.to_csv(PATH_TO_LABELS, index=False) # save updated statuses to csv file
    print('[ok] Done. Exiting...')
else:
    print('[!!] Not updating statuses. Exiting...')