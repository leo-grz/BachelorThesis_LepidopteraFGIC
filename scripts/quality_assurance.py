
# This script takes a tensor file (244x244x3) and assigns them to either blacklist or manualcheck
# it does not look at any images contained in manualcheck, blacklist or whitelist

# should a qa_checked_and_approved.txt file be used?

# Imports
import sys 
import os 
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from moth_dataset import MothDataset
from utils import show_sample

# Config
random.seed(42)

PATH_TO_PROCESSED = 'C:/Users/Leo/Desktop/BA_MothClassification/data/processed/'
PATH_TO_LABELS = PATH_TO_PROCESSED + 'testing_dataset_top20x50.csv'
PATH_TO_IMAGES = PATH_TO_PROCESSED + 'testing_dataset_top20x50_images'
#os.chmod(folder_path, 0o777)
PATH_TO_WHITELIST = PATH_TO_PROCESSED + 'qa_whitelist.txt'
PATH_TO_BLACKLIST = PATH_TO_PROCESSED + 'qa_blacklist.txt'
PATH_TO_MANUALCHECKLIST = PATH_TO_PROCESSED + 'qa_manualcheck.txt'

# Verify Config
# print('\n###\tVerifying config:\n')
# for list_file in [PATH_TO_BLACKLIST, PATH_TO_WHITELIST, PATH_TO_MANUALCHECKLIST]:
#     if os.path.exists(list_file): 
#         print(f'File found: {list_file}') 
#     else: 
#         print(f'File not found: {list_file}\t Exiting...')
#         sys.exit(1)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),          # Convert to tensor
])
full_dataset = MothDataset(csv_file=PATH_TO_LABELS, root_dir=PATH_TO_IMAGES, transform=transform)



print('\n###\Reading in Dataset:\n')
# only process files who's filenames are not already in whitelist, manualcheck or blacklist

# Function to display an image
def show_image(image, label):
    image = image.numpy().transpose((1, 2, 0))  # Convert tensor to NumPy array and transpose for display
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')
    plt.show()

dataloader = DataLoader(full_dataset, batch_size=100, shuffle=False)

for images, labels, gbifids, img_names in dataloader:
    show_sample(images[0], labels[0], full_dataset.decode_label(labels[0]), gbifids[0], img_names[0])


# Brightness check

# Sharpness check

# Size of object check



# show some statistics

