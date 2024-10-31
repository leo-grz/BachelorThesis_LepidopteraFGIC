import os
import requests
import pandas as pd
import numpy as np
import time

content_folder = 'BA_DATASETS/DS1/processed_datasetV1'
output_folder = 'BA_DATASETS/DS1/processed_datasetV1/testing_dataset'

testing_dataset = pd.read_csv(f'{content_folder}/testing_dataset.csv', low_memory=False)
multimedia = pd.read_csv(f'{content_folder}/multimedia.csv', low_memory=False)

sizes = []

def download_image(index, image_id, image_url):
    response = requests.get(image_url)

    original_filename = image_url.split('/')[-1]
    output_filename = f'{image_id}_{original_filename}'
    #output_filename = f'{image_id}.jpg'
    
    # Check if the request was successful
    if response.status_code == 200:
        # Save the image to the local directory
        with open(f'{output_folder}/{output_filename}', 'wb') as file:
            file.write(response.content)
        content_length = int(response.headers.get('Content-Length')) / 1024
        sizes.append(content_length)
        print(f"[{index}] Downloaded {output_filename} of size {content_length:.2f} kilobytes")
    else:
        print(f"[{index}] Failed to download {image_url}")
    time.sleep(0.5)


for index, row in testing_dataset.iterrows():
    image_id = row['gbifID']
    if image_id in [1966132886, 4459520332]: # only download specific gbifID-images that are in datafile and 'whitelist'
        image_url = multimedia.loc[multimedia['gbifID'] == image_id, 'identifier'].values[0]
        print(f'[{index}] {image_id} | {image_url}')
        download_image(index, image_id, image_url)

#print(f'\nAverage content length over {len(sizes)} occurences: {round(np.mean(sizes), 2)} kilobytes')






'''MULTIMEDIA.TXT
multimedia.txt head:
gbifID	type	format	identifier	references	title	description	source	audience	created	creator	contributor	publisher	license	rightsHolder
1950935083	StillImage	image/jpeg	https://observation.org/photos/8126181.jpg							Marc de Bont			http://creativecommons.org/licenses/by-nc-nd/4.0/	Marc de Bont
1950946982	StillImage	image/jpeg	https://observation.org/photos/8125216.jpg							Theo Bakker			http://creativecommons.org/licenses/by-nc/4.0/	Theo Bakker


over first 50 samples average content length of 150kb
sample size: 4.271.288
=> estimated size of 12.2 GB
at dl speed of 2 per second total dataset download time: 25days
'''


'''OCCURRENCES.TXT


'''


'''DATA CLEANING

verschiedene IDs kommem mehrfach vor (wahrscheinlich gleiches objekt)
bspw: 4 Bilder von gbifID: 1951009387

Nur Bilder zwischen 50 und 150kb?

'''