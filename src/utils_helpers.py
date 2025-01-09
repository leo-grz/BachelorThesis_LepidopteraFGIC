import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def show_sample(image, label, label_dec, gbifid, image_name, predicted=None, predicted_dec=None):
    image = image.numpy().transpose((1, 2, 0))
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')  # Turn off the axis

    # Creating the text to be displayed
    info_text = f"Labeled: {label}, {label_dec}\n"
    if predicted and predicted_dec: info_text += f"Predicted: {predicted}, {predicted_dec}\n"
    info_text += f"GBIF ID: {gbifid}\n" \
                f"Filename: {image_name}" 

    # Adding the text box
    props = dict(boxstyle='square', facecolor='lightblue', alpha=0.5)
    plt.text(1.03, 0.8, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='center', bbox=props)

    plt.show()
    plt.pause(0.001)  

def check_folder_exists(folder_path, min_fileamount=10): 
    if os.path.exists(folder_path) and os.path.isdir(folder_path): 
        listdir = os.listdir(folder_path)
        if len(listdir) < min_fileamount:
            raise FileNotFoundError(f"The folder '{folder_path}' does not contain more than {min_fileamount} files.")
    else: 
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

def load_features(filename): 
    data = np.load(filename) 
    features = data['features'] 
    labels = data['labels'] 
    gbifids = data['gbifids']
    print(f"Features and labels loaded from {filename}") 
    return features, labels, gbifids

def save_features(features, labels, gbifids, filename): 
    if os.path.exists(filename): # Load existing data 
        old_features, old_labels, old_gbifids = load_features(filename)
        features = np.concatenate((old_features, features)) 
        labels = np.concatenate((old_labels, labels)) 
        gbifids = np.concatenate((old_gbifids, gbifids)) 
    np.savez_compressed(filename, features=features, labels=labels, gbifids=gbifids) 
    print(f"Features and labels saved to {filename}") 

def get_labelmapping(df):
    species_list = sorted(df['scientificName'].unique())
    species_dict = {species: idx for idx, species in enumerate(species_list)}
    return species_dict
