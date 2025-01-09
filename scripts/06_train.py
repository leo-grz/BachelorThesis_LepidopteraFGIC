
# Imports
import torch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

import logging
import numpy as np
import os
import sys
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import check_folder_exists

# Configuration of Cross Validation and logging

PATH_TO_DATA = '/home/lgierz/BA_MothClassification/data/'
PATH_TO_FEATURES = PATH_TO_DATA + 'processed/features/'
PATH_TO_LOGFILE = PATH_TO_DATA + 'status/06_train.log'

for folder in [PATH_TO_DATA, PATH_TO_FEATURES]:
    check_folder_exists(folder, min_fileamount=0)

FOLDS = 5

KNN_PARAM_GRID = {
    'n_neighbors': [5,7,9]#,11,13,15]#,20,30],
    #'weights': ['uniform', 'distance'],
    #'algorithm': ['auto', 'ball_tree'],#, 'kd_tree', 'brute'],
    #'leaf_size': [20, 30, 40],
    #'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

SCORER = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

FOUNDATIONAL_MODELS = [
    'ResNet50_ImageNet1KV1', 
    'ResNet50_ImageNet1KV2', 
    'ResNet101_ImageNet1KV1', 
    'ResNet101_ImageNet1KV1'
]

DATASETS = [
    'top277_max3000',
    'top277_max2000',
    'top277_max1000',
    'top277_max500',
    'top387_max2000',
    'top387_max1000',
    'top387_max500',
    'top589_max1000',
    'top589_max500',
]

DATASET_NAME = DATASETS[0] # top277_max3000
MODEL_NAME = FOUNDATIONAL_MODELS[1] # resnet50 imagenet v2

# Configure logging
logging.basicConfig(
    filename=PATH_TO_LOGFILE,
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
)
logger = logging.getLogger()


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'Device chosen: {device}')
# logging.info(f'[INIT] Device chosen: {device}')

def log_grid_search(search, params, score, n_split): 
    logger.info(f"Params: {params}, Score: {score}")


def load_features(filename): 
    data = np.load(filename) 
    features = data['features'] 
    labels = data['labels'] 
    print(f"Features and labels loaded from {filename}") 
    return features, labels


PATH_TO_FEATURES_CV_TRAIN_VAL = PATH_TO_FEATURES + f'cv_train_val_features_{MODEL_NAME}_{DATASET_NAME}.npz'

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
knn = KNeighborsClassifier()



grid_search = GridSearchCV(estimator=knn, param_grid=KNN_PARAM_GRID, scoring='accuracy', cv=skf, verbose=3, n_jobs=4)
loaded_features, loaded_labels = load_features(PATH_TO_FEATURES_CV_TRAIN_VAL)


grid_search.fit(loaded_features, loaded_labels)

print(f'Cross Validation performed with {len(loaded_features)} features and {len(loaded_labels)} labels.')


best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score:.4f}")


# Evaluate the KNN classifier on the validation set using the best estimator found by GridSearchCV

PATH_TO_FEATURES_TEST = PATH_TO_FEATURES + f'test_features_{MODEL_NAME}_{DATASET_NAME}.npz'

print(f'Final test on best parameters performed with {len(test_features)} features and {len(test_labels)} labels.')
loaded_features, loaded_labels = load_features(PATH_TO_FEATURES_TEST)

test_predictions = grid_search.best_estimator_.predict(loaded_features)
accuracy = accuracy_score(loaded_labels, test_predictions)
print(f"Testing Accuracy: {accuracy:.4f}")
