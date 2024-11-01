import sys 
import os 
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from moth_dataset import MothDataset

# Config

PATH_TO_PROCESSED = 'C:/Users/Leo/Desktop/BA_MothClassification/data/processed/'
PATH_TO_LABELS = PATH_TO_PROCESSED + 'testing_dataset_top20x50.csv'
PATH_TO_IMAGES = PATH_TO_PROCESSED + 'testing_dataset_top20x50_images'


# prepare rows to contain CHECK to be inspected manually
csv_file = pd.read_csv(PATH_TO_LABELS)
csv_file['status'] = csv_file['status'].astype('str')
csv_file_filtered = csv_file[csv_file['status'] == 'CHECK']
csv_file_for_loader = csv_file_filtered[['gbifID', 'scientificName']]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),          # Convert to tensor
])

check_dataset = MothDataset(csv_file=csv_file_for_loader, root_dir=PATH_TO_IMAGES, transform=transform)
dataloader = DataLoader(check_dataset, batch_size=25, shuffle=False)

whitelist = []
blacklist = []

# Events for setting status 
def on_key(event, gbifid): 
    if event.key == 'b': 
        print(f"{gbifid} -> BLACKLISTED")
        blacklist.append(int(gbifid))
        plt.close(event.canvas.figure)
    elif event.key == 'w':
        print(f"{gbifid} -> WHITELISTED")
        whitelist.append(int(gbifid))
        plt.close(event.canvas.figure)

def categorize_show_sample(image, img_name, gbifid):
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

for gbifid in whitelist:
    csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'WHITE'

for gbifid in blacklist:
    csv_file.loc[csv_file['gbifID'] == gbifid, 'status'] = 'BLACK'

csv_file.to_csv(PATH_TO_LABELS, index=False)


