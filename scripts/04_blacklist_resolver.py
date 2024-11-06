'''
This file takes the dataset and appends new unseen samples of the classes corresponding to the blacklisted samples. 
It doesnt delete or change anything, just adds samples to the labels with status=NEW.

It is executed after quality_assurance.py and manual_checker.py
but those scripts should be executed again to check the newly added samples (status=NEW)
'''

# Imports
import pandas as pd

PATH_TO_DATA = 'C:/Users/Leo/Desktop/BA_MothClassification/data/'
PATH_TO_DS_SLICE = PATH_TO_DATA + 'processed/testing_dataset_top20_max50.csv'
PATH_TO_IMAGES = PATH_TO_DATA + 'processed/testing_dataset_top20_max50_images'
PATH_TO_DS_FULL = PATH_TO_DATA + 'processed/dataset_full.csv'

slice_df = pd.read_csv(PATH_TO_DS_SLICE)
slice_df['status'] = slice_df['status'].astype('str')

full_df = pd.read_csv(PATH_TO_DS_FULL)

known_gbifids = slice_df['gbifID'].tolist() # Create blacklist of gbifIDs not to include from the large dataset
unseen_samples = full_df[~full_df['gbifID'].isin(known_gbifids)] # Create dataset with unknown samples only
black_samples = slice_df[slice_df['status'] == 'BLACK'] # Select all blacklisted samples
target_classes = black_samples['scientificName'].value_counts() # Get scientificName values of blacklisted samples and their counts

selected_samples = pd.DataFrame() # Create an empty DataFrame to store the selected samples

# Sample the larger dataset for each scientificName in target_classes
for name, count in target_classes.items():
    class_samples = unseen_samples[unseen_samples['scientificName'] == name]
    print(f'Trying to select {count} samples for {name}. Available: {len(class_samples)}')

    sampled_class = class_samples.sample(n=min(len(class_samples), count)) # try to select wanted amount of images per target class
    selected_samples = pd.concat([selected_samples, sampled_class]) # 

selected_samples['status'] = 'NEW'

combined_df = pd.concat([slice_df, selected_samples], ignore_index=True)
combined_df = combined_df[['gbifID', 'scientificName', 'identifier', 'status']]

combined_df.to_csv(PATH_TO_DS_SLICE, index=False)

# alle dl-fails werden im original file geblacklisted ????????????
