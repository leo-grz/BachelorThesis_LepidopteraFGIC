import time
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os
import logging
import torch
from torch import nn
import numpy as np
import pandas as pd
from time import perf_counter as pc
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils_helpers import check_folder_exists, load_features


# Paths
PATH_TO_DATA = '/home/lgierz/BA_MothClassification/data/'
PATH_TO_DATASETS = PATH_TO_DATA + 'processed/cv_datasets/'
PATH_TO_RESULTS = PATH_TO_DATA + 'processed/results/'

for path in [PATH_TO_DATA, PATH_TO_DATASETS, PATH_TO_RESULTS]:
    check_folder_exists(path, 0)

FOLDS = 10
FOUNDATIONAL_MODEL_NAMES = ['dino']#['resnet', 'dino']
MODEL_NAMES = ["KNN", "Linear Classifier"]
COMPONENTS = 512

# DATASET_CONFIGS = {
#     'top277': (277, [3000, 2000, 1000, 500]),
#     'top387': (387, [2000, 1000, 500]),
#     'top589': (589, [1000, 500])
# }

DATASET_CONFIGS = {
    'top277': (277, [2000, 1000, 500]),
    'top387': (387, [2000, 1000, 500]),
    'top589': (589, [1000, 500])
}

MODEL_CONFIGS = {
    'linear__learning_rate': [0.005, 0.001, 0.0005],
    'linear__epochs': [1000, 1500, 2000],

    'knn__neighbors': [35, 50, 65],
}

results_template = {
    "Model": None,
    "Reduction Time (s)": None,
    "Training Time (s)": None,
    "Accuracy": None,
    "Precision": None,
    "Recall": None,
    "F1-Score": None,
    "Neighbors": None,
    "Learning Rate": None,
    "Epochs": None,
    "Epoch Losses": None,
    "Epoch Accuracies": None
}

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Device configuration for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def handle_results(csv_path, results):
    """
    Handle the results by calculating metrics and saving them to a CSV file.
    """
    results_df = pd.DataFrame(results)
    if not os.path.isfile(csv_path):
        results_df.to_csv(csv_path, index=False)
    else:
        results_df.to_csv(csv_path, mode='a', header=False, index=False)

    logging.info(f"Results saved to {csv_path}")

# PyTorch Linear Classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)



for fm in FOUNDATIONAL_MODEL_NAMES:

    # Iterate through dataset configurations
    for dataset_name, (class_amount, sample_amounts) in DATASET_CONFIGS.items():
        
        # Iterate through the sample amounts
        for sample_amount in sample_amounts:

            feature_file = PATH_TO_DATASETS + f'{fm}_feature_dataset_top{class_amount}_max{sample_amount}.npz'
            logfile = PATH_TO_RESULTS + f'resnet_cv_test_top277_max3000.log'
            results_file = PATH_TO_RESULTS + f'{fm}_cv_test_top{class_amount}_max{sample_amount}.csv'

            logging.basicConfig(
                filename=logfile,
                level=logging.INFO,
                format='[%(asctime)s][%(levelname)s] - %(message)s',
            )

            # Logging handler setup
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s'))
            logger = logging.getLogger()
            logger.addHandler(console_handler)

            proceed = 'y'#input(f'About to load features from {feature_file}... proceed? y/n ')
            if proceed.lower() != 'y':
                sys.exit(0)

            # Loading features
            features, labels, _ = load_features(feature_file)

            # PCA Reduction
            start_time = pc()
            reducer = PCA(n_components=COMPONENTS, random_state=42)
            reduced_features = reducer.fit_transform(features)
            logging.info(f'Loaded features with shape {features.shape}, reduced to {reduced_features.shape} using PCA')
            reduction_time = pc() - start_time



            for model_name in MODEL_NAMES:
                if model_name == "Linear Classifier":
                    for epochs in MODEL_CONFIGS['linear__epochs']:
                        for lr in MODEL_CONFIGS['linear__learning_rate']:
                            fold_results = []
                            avg_results = results_template.copy()

                            for fold, (train_index, test_index) in enumerate(skf.split(reduced_features, labels), start=1):
                                X_train, X_test = reduced_features[train_index], reduced_features[test_index]
                                y_train, y_test = labels[train_index], labels[test_index]
                                
                                if fold == 1: logging.info(f'[BEFORE LABEL CORRECTION] Lowest label: {np.min(y_train)}, highest label: {np.max(y_train)}, unique labels in training ds: {len(np.unique(y_train))}, unique labels in testing ds: {len(np.unique(y_test))}')


                                # Fix missing labels due to dataset splitting by changing label to ascending order
                                label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
                                y_train = np.array([label_mapping[label] for label in y_train])
                                y_test = np.array([label_mapping[label] for label in y_test])

                                if fold == 1: logging.info(f'[AFTER LABEL CORRECTION] Lowest label: {np.min(y_train)}, highest label: {np.max(y_train)}, unique labels in training ds: {len(np.unique(y_train))}, unique labels in testing ds: {len(np.unique(y_test))}')
                                logging.info(f'[FOLD {fold}] Shapes: {X_train.shape} (X training) | {X_test.shape} (X testing) | {y_train.shape} (y training) | {y_test.shape} (y testing)')

                                losses, accuracies = [], []

                                # PyTorch Model Setup
                                input_dim = X_train.shape[1] # TODO: learning rate schedulinng!!

                                start_time = pc()

                                linear_model = LinearClassifier(input_dim, class_amount).to(device)
                                criterion = nn.CrossEntropyLoss()
                                optimizer = torch.optim.Adam(linear_model.parameters(), lr=lr)
                                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                                y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

                                for epoch in range(epochs):
                                    linear_model.train()
                                    optimizer.zero_grad()

                                    outputs = linear_model(X_train_tensor)
                                    loss = criterion(outputs, y_train_tensor)
                                    loss.backward()
                                    optimizer.step()

                                    # Calculate accuracy
                                    _, predicted = torch.max(outputs.data, 1)
                                    correct = (predicted == y_train_tensor).sum().item()
                                    accuracy = correct / y_train_tensor.size(0)

                                    losses.append(round(loss.item(), 4))
                                    accuracies.append(round(accuracy, 4))

                                # Evaluate Linear Classifier
                                linear_model.eval()
                                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                                y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
                                with torch.no_grad():
                                    outputs = linear_model(X_test_tensor)
                                    _, y_pred = torch.max(outputs, 1)
                                    y_pred_numpy = y_pred.cpu().numpy()

                                training_time = pc() - start_time

                                fold_results.append({
                                    "Model": model_name,
                                    "Reduction Time (s)": reduction_time,
                                    "Training Time (s)": training_time,
                                    "Accuracy": accuracy_score(y_test, y_pred_numpy),
                                    "Precision": precision_score(y_test, y_pred_numpy, average='weighted'),
                                    "Recall": recall_score(y_test, y_pred_numpy, average='weighted'),
                                    "F1-Score": f1_score(y_test, y_pred_numpy, average='weighted'),
                                    "Neighbors": None,
                                    "Learning Rate": lr,
                                    "Epochs": epochs,
                                    "Epoch Losses": losses,
                                    "Epoch Accuracies": accuracies
                                })

                            # Calculate average results over folds
                            for key in avg_results.keys():
                                if key not in ["Model", "Neighbors", "Epoch Losses", "Epoch Accuracies"]:
                                    avg_results[key] = round(np.mean([fold[key] for fold in fold_results if isinstance(fold[key], (int, float))]), 4)
                                else: 
                                    avg_results[key] = fold_results[0][key]

                            avg_results["Epoch Losses"] = np.mean([fold["Epoch Losses"] for fold in fold_results], axis=0).tolist()
                            avg_results["Epoch Losses"] = [round(loss, 4) for loss in avg_results["Epoch Losses"]]
                            avg_results["Epoch Accuracies"] = np.mean([fold["Epoch Accuracies"] for fold in fold_results], axis=0).tolist()
                            avg_results["Epoch Accuracies"] = [round(acc, 4) for acc in avg_results["Epoch Accuracies"]]
                            handle_results(results_file, [avg_results])

                elif model_name == 'KNN':
                    for neighbors in MODEL_CONFIGS['knn__neighbors']:
                        fold_results = []
                        avg_results = results_template.copy()

                        for fold, (train_index, test_index) in enumerate(skf.split(reduced_features, labels), start=1):
                            X_train, X_test = reduced_features[train_index], reduced_features[test_index]
                            y_train, y_test = labels[train_index], labels[test_index]


                            if fold == 1: logging.info(f'[BEFORE LABEL CORRECTION] Lowest label: {np.min(y_train)}, highest label: {np.max(y_train)}, unique labels in training ds: {len(np.unique(y_train))}, unique labels in testing ds: {len(np.unique(y_test))}')
                            # Fix missing labels due to dataset splitting by changing label to ascending order
                            label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
                            y_train = np.array([label_mapping[label] for label in y_train])
                            y_test = np.array([label_mapping[label] for label in y_test])

                            if fold == 1: logging.info(f'[AFTER LABEL CORRECTION] Lowest label: {np.min(y_train)}, highest label: {np.max(y_train)}, unique labels in training ds: {len(np.unique(y_train))}, unique labels in testing ds: {len(np.unique(y_test))}')
                            logging.info(f'[FOLD {fold}] Shapes: {X_train.shape} (X training) | {X_test.shape} (X testing) | {y_train.shape} (y training) | {y_test.shape} (y testing)')


                            start_time = pc() 

                            model = KNeighborsClassifier(n_neighbors=neighbors)
                            model.fit(X_train, y_train)
                            y_pred_numpy = model.predict(X_test)# predictions hohe confidance?, gehören nachbarn zur gleichen klasse?
                            # Sicherheit herausfinden, wie hoch ist die confidance für einzelne samples und für die verschiedenen klassen
                            # wie hoch ist die confidance, funktion herausfinden (sklearn), umd die confidance von einzelnen samples herauszufinden
                            training_time = pc() - start_time 

                            fold_results.append({
                                "Model": model_name,
                                "Reduction Time (s)": reduction_time,
                                "Training Time (s)": training_time,
                                "Accuracy": accuracy_score(y_test, y_pred_numpy),
                                "Precision": precision_score(y_test, y_pred_numpy, average='weighted'),
                                "Recall": recall_score(y_test, y_pred_numpy, average='weighted'),
                                "F1-Score": f1_score(y_test, y_pred_numpy, average='weighted'),
                                "Neighbors": neighbors,
                                "Learning Rate": None,
                                "Epochs": None,
                                "Epoch Losses": None,
                                "Epoch Accuracies": None
                            })

                        # Calculate average results over folds
                        for key in avg_results.keys():
                            if key not in ["Model", "Epochs", "Learning Rate", "Epoch Losses", "Epoch Accuracies"]:
                                avg_results[key] = round(np.mean([fold[key] for fold in fold_results if isinstance(fold[key], (int, float))]), 4)
                            else:
                                avg_results[key] = fold_results[0][key]

                        handle_results(results_file, [avg_results])
            
            gc.collect()
            torch.cuda.empty_cache()


print('Done!')