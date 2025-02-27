import time
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
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

FOLDS = 5
FOUNDATIONAL_MODEL_NAMES = ['dino', 'resnet']
MODEL_NAMES = ["Linear Classifier", 'KNN']
COMPONENTS = 512 # <---------------------- PCA COMPONENTS


DATASET_CONFIGS = {
    'top277': (277, [500, 1000, 2000,3000]),
    'top387': (387, [500, 1000, 2000]),
    'top589': (589, [500,1000])
}

MODEL_CONFIGS_DINO = {
    'linear__learning_rate': [0.0005, 0.0001, 0.00005],
    'linear__epochs': [1500, 2000],
    'knn__neighbors': [20, 35, 50],
}

MODEL_CONFIGS_RESNET = {
    'linear__learning_rate': [0.005, 0.001, 0.0005],
    'linear__epochs': [1500, 2000],
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
    "Validation Losses": None,
    "Validation Accuracies": None
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

    config = MODEL_CONFIGS_DINO if fm == 'dino' else MODEL_CONFIGS_RESNET

    # Iterate through dataset configurations
    for dataset_name, (class_amount, sample_amounts) in DATASET_CONFIGS.items():
        
        # Iterate through the sample amounts
        for sample_amount in sample_amounts:

            time.sleep(60)


            feature_file = PATH_TO_DATASETS + f'{fm}_feature_dataset_top{class_amount}_max{sample_amount}.npz'
            logfile = PATH_TO_RESULTS + f'dino_cross_validation_experiment.log'
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
                    for epochs in config['linear__epochs']:
                        for lr in config['linear__learning_rate']:
                            start = pc()
                            gc.collect()
                            torch.cuda.empty_cache()
                            print('Cache was cleared, took ',pc() - start)

                            fold_results = []
                            avg_results = results_template.copy()

                            for fold, (train_index, test_index) in enumerate(skf.split(reduced_features, labels), start=1):
                                X_train, X_temp = reduced_features[train_index], reduced_features[test_index]
                                y_train, y_temp = labels[train_index], labels[test_index]

                                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

                                if fold == 1: logging.info(f'[BEFORE LABEL CORRECTION] Lowest label: {np.min(y_train)}, highest label: {np.max(y_train)}, unique labels in training ds: {len(np.unique(y_train))}, unique labels in testing ds: {len(np.unique(y_test))}')


                                # Fix missing labels due to dataset splitting by changing label to ascending order
                                label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
                                y_train = np.array([label_mapping[label] for label in y_train])
                                y_val = np.array([label_mapping[label] for label in y_val])
                                y_test = np.array([label_mapping[label] for label in y_test])

                                if fold == 1: logging.info(f'[AFTER LABEL CORRECTION] Lowest label: {np.min(y_train)}, highest label: {np.max(y_train)}, unique labels in training ds: {len(np.unique(y_train))}, unique labels in testing ds: {len(np.unique(y_test))}')
                                logging.info(f'[FOLD {fold}] Shapes: {X_train.shape} (X training) | {X_val.shape} (X validation) | {X_test.shape} (X testing) | {y_train.shape} (y training) | {y_val.shape} (y validation) | {y_test.shape} (y testing)')

                                val_losses, val_accuracies = [], []

                                # PyTorch Model Setup
                                input_dim = X_train.shape[1]

                                start_time = pc()

                                linear_model = LinearClassifier(input_dim, class_amount).to(device)
                                criterion = nn.CrossEntropyLoss()
                                optimizer = torch.optim.Adam(linear_model.parameters(), lr=lr)
                                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                                y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
                                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                                y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

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

                                    # Validation accuracy and loss
                                    linear_model.eval()
                                    with torch.no_grad():
                                        val_outputs = linear_model(X_val_tensor)
                                        val_loss = criterion(val_outputs, y_val_tensor)
                                        _, val_predicted = torch.max(val_outputs.data, 1)
                                        val_correct = (val_predicted == y_val_tensor).sum().item()
                                        val_accuracy = val_correct / y_val_tensor.size(0)
                                        val_losses.append(round(val_loss.item(), 4))
                                        val_accuracies.append(round(val_accuracy, 4))

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
                                    "Validation Accuracies": val_accuracies,
                                    "Validation Losses": val_losses
                                })

                            # Calculate average results over folds
                            for key in avg_results.keys():
                                if key not in ["Model", "Neighbors", "Validation Losses", "Validation Accuracies"]:
                                    avg_results[key] = round(np.mean([fold[key] for fold in fold_results if isinstance(fold[key], (int, float))]), 4)
                                else: 
                                    avg_results[key] = fold_results[0][key]


                            avg_results["Validation Losses"] = np.mean([fold["Validation Losses"] for fold in fold_results], axis=0).tolist()
                            avg_results["Validation Losses"] = [round(loss, 4) for loss in avg_results["Validation Losses"]]
                            avg_results["Validation Accuracies"] = np.mean([fold["Validation Accuracies"] for fold in fold_results], axis=0).tolist()
                            avg_results["Validation Accuracies"] = [round(acc, 4) for acc in avg_results["Validation Accuracies"]]
                            handle_results(results_file, [avg_results])

                elif model_name == 'KNN':
                    for neighbors in config['knn__neighbors']:
                        start = pc()
                        gc.collect()
                        torch.cuda.empty_cache()
                        print('Cache was cleared, took ',pc() - start)
                        
                        fold_results = []
                        avg_results = results_template.copy()

                        for fold, (train_index, test_index) in enumerate(skf.split(reduced_features, labels), start=1):
                            X_train, X_temp = reduced_features[train_index], reduced_features[test_index]
                            y_train, y_temp = labels[train_index], labels[test_index]

                            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

                            if fold == 1: logging.info(f'[BEFORE LABEL CORRECTION] Lowest label: {np.min(y_train)}, highest label: {np.max(y_train)}, unique labels in training ds: {len(np.unique(y_train))}, unique labels in testing ds: {len(np.unique(y_test))}')
                            # Fix missing labels due to dataset splitting by changing label to ascending order
                            label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
                            y_train = np.array([label_mapping[label] for label in y_train])
                            y_val = np.array([label_mapping[label] for label in y_val])
                            y_test = np.array([label_mapping[label] for label in y_test])

                            if fold == 1: logging.info(f'[AFTER LABEL CORRECTION] Lowest label: {np.min(y_train)}, highest label: {np.max(y_train)}, unique labels in training ds: {len(np.unique(y_train))}, unique labels in testing ds: {len(np.unique(y_test))}')
                            logging.info(f'[FOLD {fold}] Shapes: {X_train.shape} (X training) | {X_val.shape} (X validation) | {X_test.shape} (X testing) | {y_train.shape} (y training) | {y_val.shape} (y validation) | {y_test.shape} (y testing)')

                            start_time = pc() 

                            model = KNeighborsClassifier(n_neighbors=neighbors)
                            model.fit(X_train, y_train)
                            y_pred_numpy = model.predict(X_test)
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
                                "Validation Accuracies": None,
                                "Validation Losses": None
                            })

                        # Calculate average results over folds
                        for key in avg_results.keys():
                            if key not in ["Model", "Epochs", "Learning Rate", "Validation Accuracies", "Validation Losses"]:
                                avg_results[key] = round(np.mean([fold[key] for fold in fold_results if isinstance(fold[key], (int, float))]), 4)
                            else:
                                avg_results[key] = fold_results[0][key]

                        handle_results(results_file, [avg_results])
            

print('Done!')