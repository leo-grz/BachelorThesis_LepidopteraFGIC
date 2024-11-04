'''
This script is used to download the 1.8 million sample dataset from occurrences.org. The estimated size 
of the download is ca. 270 GB and will take about 7 days at a download speed of two images per second.
'''

# Imports
import csv
import requests
import pandas as pd
import numpy as np
import time
import logging

# Config 

PATH_TO_PROCESSED = 'C:/Users/Leo/Desktop/BA_MothClassification/data/processed/'
PATH_TO_LABELS = PATH_TO_PROCESSED + 'dataset_top589_max5000_TESTING.csv'

PATH_TO_CURRENT = 'C:/Users/Leo/Desktop/BA_MothClassification/scripts/download_test/'
PATH_TO_OUTPUT_FOLDER = PATH_TO_CURRENT + 'images/' # /mnt/data/some_folder
PATH_TO_LOGFILE = PATH_TO_CURRENT + 'download.log'
PATH_TO_FAILED_DL_CSV = PATH_TO_CURRENT + 'failed_dl_of_top589_max5000.csv'

SIZE_AVG_RESET = 10 # amount of downloads after an Average should be printed to logfile
SLEEPING_DURATION = 0.5

# Configure logging
logging.basicConfig(
    filename=PATH_TO_LOGFILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

dataset = pd.read_csv(PATH_TO_LABELS, low_memory=False)

sizes = []
averages = []

dl_attempts = 0 # total dl counter
dl_fails = 0 # init dl fail counter

dl_attempts_batch = 0
dl_fails_batch = 0

def append_row_to_csv(file_path, row): 
    with open(file_path, 'a', newline='') as file: 
        writer = csv.writer(file) 
        writer.writerow(row)

def download_image(image_id, image_url):

    response = requests.get(image_url)
    original_filename = image_url.split('/')[-1]
    output_filename = f'{image_id}_{original_filename}'    
    
    if response.status_code == 200: # Check if the request was successful
        # Save the image to the local directory
        with open(f'{PATH_TO_OUTPUT_FOLDER}/{output_filename}', 'wb') as file:
            file.write(response.content)
        content_length = int(response.headers.get('Content-Length')) / 1024
        sizes.append(content_length)
        return 0
    else:
        return 1
try:
    used_dataset = dataset[0:50]

    for index, row in used_dataset.iterrows():
        image_id = row['gbifID']
        image_url = row['identifier']
        image_label = row['scientificName']
        status = download_image(image_id, image_url)
        if status == 1:
            dl_fails += 1
            dl_fails_batch += 1

            row = [image_id, image_label, image_url]
            append_row_to_csv(PATH_TO_FAILED_DL_CSV, row)
            logging.error(f'[{index}][FAIL] ID: {image_id} | URL: {image_url}')
            print(f'[{index}][FAIL] ID: {image_id} | URL: {image_url}')
        else:
            logging.info(f'[{index}][SUCCESS] ID: {image_id} | URL: {image_url}')
            print(f'[{index}][SUCCESS] ID: {image_id} | URL: {image_url}')

        dl_attempts += 1
        dl_attempts_batch += 1

        if dl_attempts % SIZE_AVG_RESET == 0:
            avg = round(np.mean(sizes), 2) # average size in kb over last 1000 samples
            averages.append(avg)
            logging.info(f'[REPORT] Average content length over {len(sizes)} occurences: {avg} kilobytes')
            logging.info(f'[REPORT] BATCH: [FAILED/ATTEMPTED] -> [{dl_fails_batch}/{dl_attempts_batch}]')
            logging.info(f'[REPORT] TOTAL: [FAILED/ATTEMPTED] -> [{dl_fails}/{dl_attempts}]')
            sizes = []
            dl_attempts_batch = 0
            dl_fails_batch = 0

        time.sleep(SLEEPING_DURATION)

    logging.info(f'[FINISHED] Estimated size according to averages: {round(sum(averages) * SIZE_AVG_RESET / (1024**2), 4)} GB')
    logging.info(f'[FINISHED] Amount of failed download attempts: {dl_fails}')

except KeyboardInterrupt: 
    logging.critical("KeyboardInterrupt detected and handled.") 
    print("Process interrupted by user.")

except Exception as e:
    logging.critical(f'An Error occurred: {e}')

