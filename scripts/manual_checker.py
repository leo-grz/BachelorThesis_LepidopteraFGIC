# in this script a GUI will be implemented to assign the images 
# from manualcheck.txt to either black- or whitelist

import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from moth_dataset import MothDataset
from utils import show_sample