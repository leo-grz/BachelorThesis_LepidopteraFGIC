import csv
import os

'''
Script to check if a sample image exists for a given gbifID in the data set (csv)
'''


def check_files(csv_path, folder_path):
    missing_files = []

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        
        for row in reader:
            id = row[0]  # Assuming the ID is in the first column of each row
            # Check if a file starting with the ID exists in the folder
            files = [f for f in os.listdir(folder_path) if f.startswith(id)]
            if not files:
                missing_files.append(id)
                print(f"No file found for ID {id}")

    if not missing_files:
        print("All files found.")
    else:
        print(f"Missing files for IDs: {missing_files}")

# Example usage
csv_path = 'C:/Users/Leo/Documents/Drive/BA/code/BA_DATASETS/DS1/processed_datasetV1/testing_dataset.csv'
folder_path = 'C:/Users/Leo/Documents/Drive/BA/code/BA_DATASETS/DS1/processed_datasetV1/testing_dataset'
check_files(csv_path, folder_path)
