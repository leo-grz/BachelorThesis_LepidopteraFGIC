
# Imports
import time 
import sys 
import os 
import numpy as np
import random

# Config
random.seed(42)

PATH_TO_IMAGES = None
SAVE_TENSORS_FILE = True

PATH_TO_EXISTING_TENSORS_FILE = None

PATH_TO_WHITELIST = ''
PATH_TO_BLACKLIST = ''
PATH_TO_MANUALCHECKLIST = ''

# Verify Config
print('\n###\tVerifying config:\n')
for list_file in [PATH_TO_BLACKLIST, PATH_TO_WHITELIST, PATH_TO_MANUALCHECKLIST]:
    if os.path.exists(list_file): 
        print(f'File found: {list_file}') 
    else: 
        print(f'File not found: {list_file}\t Exiting...')
        sys.exit(1)

if not PATH_TO_IMAGES and not PATH_TO_EXISTING_TENSORS_FILE:
    print('Neigher path to images nor path to a tensors-file is given! Exiting...')
    sys.exit(1)



print('\n###\Reading in Dataset:\n')

if not PATH_TO_EXISTING_TENSORS_FILE:

    dataset = ''
else:
    dataset = ''


# Read in Dataset and turn them into 224x224 RGB Tensors. Optional to save the file

# Brightness check

# Sharpness check

# Size of object check



# show some statistics

