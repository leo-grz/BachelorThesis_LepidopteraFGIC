'''
This script determines, if samples registered for a manual check (status = CHECK) by quality_assurance.py 
should be black- or whitelisted.
This script should be executed before quality_assurance.py and blacklist_resolver.py
'''

# Imports
import sys 
import os 
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from lepidoptera_dataset import LepidopteraDataset

# Config

PATH_TO_DATA = 'C:/Users/Leo/Desktop/BA_MothClassification/data/'
PATH_TO_LABELS = PATH_TO_DATA + 'processed/testing_dataset_top20_max50.csv'
PATH_TO_IMAGES = PATH_TO_DATA + 'processed/testing_dataset_top20_max50_images'

csv_file = pd.read_csv(PATH_TO_LABELS)
csv_file['status'] = csv_file['status'].astype('str')
csv_file_filtered = csv_file[csv_file['status'] == 'CHECK'] # select all samples with status CHECK
csv_file_for_loader = csv_file_filtered[['gbifID', 'scientificName']] # forward only relevant columns with status CHECK

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),          # Convert to tensor
])

check_dataset = LepidopteraDataset(csv_file=csv_file_for_loader, root_dir=PATH_TO_IMAGES, transform=transform)
dataloader = DataLoader(check_dataset, batch_size=25, shuffle=False)

whitelist = []
blacklist = []

def on_key(event, gbifid):
    """
    Handle key press events to categorize images based on the pressed key.

    This function handles key press events to categorize an image as either 
    blacklisted or whitelisted based on the key pressed by the user.

    Parameters:
    event (matplotlib.backend_bases.KeyEvent): The event object containing information about the key press.
    gbifid (int): The GBIF ID of the image being categorized.

    Actions:
    - Press 'b' to blacklist the image.
    - Press 'w' to whitelist the image.

    Notes:
    The function appends the GBIF ID to the appropriate list (blacklist or whitelist) and closes the image window.

    Returns:
    None
    """
    if event.key == 'b':
        print(f'{gbifid} -> BLACKLISTED')
        blacklist.append(int(gbifid))
        plt.close(event.canvas.figure)
    elif event.key == 'w':
        print(f'{gbifid} -> WHITELISTED')
        whitelist.append(int(gbifid))
        plt.close(event.canvas.figure)

def categorize_show_sample(image, img_name, gbifid):
    """
    Display an image and categorize it based on user key press events.

    This function displays an image using Matplotlib and waits for the user to press a key 
    to categorize the image. The key press events are handled by the on_key function.

    Parameters:
    image (torch.Tensor): A tensor of shape (3, H, W) representing an RGB image.
    img_name (str): The name of the image to be displayed in the title.
    gbifid (int): The GBIF ID of the image being displayed.

    Actions:
    - Press 'b' to blacklist the image.
    - Press 'w' to whitelist the image.

    Returns:
    None
    """
    image = image.numpy().transpose((1, 2, 0))
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(img_name)
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, gbifid))
    plt.show()

i = 0
for images, labels, gbifids, img_names in dataloader:
    i +=1
    print(f'Batch [{i}/{len(dataloader)}]')
    
    for image, label, gbifid, img_name in zip(images, labels, gbifids, img_names):

        categorize_show_sample(image, img_name, gbifid)

for gbifid in whitelist: # write WHITE status to csv file
    csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'WHITE'

for gbifid in blacklist: # write BLACK status to csv file, those will be deleted and eventually replaced in blacklist_resolver.py
    csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'BLACK'

csv_file.to_csv(PATH_TO_LABELS, index=False) # save updated statuses to csv file