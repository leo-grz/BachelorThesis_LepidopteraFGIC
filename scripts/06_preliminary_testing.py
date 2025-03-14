'''
This script performs preliminary testing of different models (Logistic Regression, KNN) 
on extracted features. It includes the feature reduction techniques PCA and UMAP, 
and evaluates the models using various metrics such as accuracy, precision, recall, and F1-score.

The script should be executed after feature extraction and is used to find the best hyperparameters
for the final cross validation test.

For this the preliminary test, the dataset is used comprising 277 classes and 1000 samples per class.
The test is conducted for both DINOv2- and ResNet-50-based datasets.
'''

# Imports
import time
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os
import logging
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils_helpers import check_folder_exists, load_features, save_features


PATH_TO_DATA = '/home/lgierz/BA_MothClassification/data/'
PATH_TO_LOGFILE = PATH_TO_DATA + 'status/hyperparameter_tuning_dino_chunky_val.log'
feature_file = PATH_TO_DATA + 'processed/cv_datasets/dino_feature_dataset_top277_max1000.npz' 
csv_file_path = PATH_TO_DATA + 'status/hyperparameter_tuning_dino_chunky_val.csv'

model_names = ["KNN", "Linear Classifier"]

config = {
    'pca__reduced_fe_size': [256, 512, 1024],
    
    'umap__reduced_fe_size': [128,512,1024],
    'umap__neighbors': [50,100],

    'knn__neighbors': [1,3,5,10,50,100,300,500,700,1000],

    'linear__learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'linear__epochs': [250,500,750,1000,1500],

    # 'linear__patience': [3,5,10],
    # 'linear__gamma': [0.2, 0.5, 0.8, 0.95]
}

logging.basicConfig(
    filename=PATH_TO_LOGFILE,
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
)

console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')) 
logger = logging.getLogger() 
logger.addHandler(console_handler)

features, labels, _ = load_features(feature_file)

X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f'Lowest label: {np.min(y_train)}, highest label: {np.max(y_train)}, unique labels in training ds: {len(np.unique(y_train))}, unique labels in testing ds: {len(np.unique(y_test))}')

# fix missing labels due to dataset splitting by changing label to ascending order
label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
y_train = np.array([label_mapping[label] for label in y_train]) # since labels are not in ascending order, remapping is necessary
y_val = np.array([label_mapping[label] for label in y_val])
y_test = np.array([label_mapping[label] for label in y_test])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def handle_results(csv_path, test, pred, method, model, params, reduction_time, training_time, neighbors=None, C=None, lr=None, epochs=None, val_losses=None, val_accuracies=None, gamma=None, patience=None):
    acc = accuracy_score(test, pred)
    prec = precision_score(test, pred, average='weighted')
    rec = recall_score(test, pred, average='weighted')
    f1 = f1_score(test, pred, average='weighted')

    # Store Results
    run_results = {
        "Method": method,
        "Model": model,
        "Parameters": params,
        "Reduction Time (s)": round(reduction_time, 2),
        "Training Time (s)": round(training_time, 2),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Neighbors": neighbors if model == 'KNN' else None,
        "C": C if model == 'SVM' else None,
        "Learning Rate": lr if model == "Linear Classifier" else None,
        "Epochs": epochs if model == "Linear Classifier" else None,
        "Gamma": gamma if model == "Linear Classifier" else None,
        "Patience": patience if model == "Linear Classifier" else None,
        "Validation Accuracies": val_accuracies if model == "Linear Classifier" else None,
        "Validation Losses": val_losses if model == "Linear Classifier" else None
    }
    # Convert the dictionary to a DataFrame
    results_df = pd.DataFrame([run_results])  # Wrap in a list to create a DataFrame with one row
    if not os.path.isfile(csv_path):
        results_df.to_csv(csv_path, index=False)
    else:
        results_df.to_csv(csv_path, mode='a', header=False, index=False)

    logging.info(f"[{run_results['Method']} ({run_results['Parameters']})][{run_results['Model']}] Reduction time: {run_results['Reduction Time (s)']}s, Training time: {run_results['Training Time (s)']}s, Accuracy: {run_results['Accuracy']}, Precision: {run_results['Precision']}, Recall: {run_results['Recall']}, F1-Score: {run_results['F1-Score']}")


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes): 
        super(LinearClassifier, self).__init__() 
        self.fc = nn.Linear(input_dim, num_classes) 
        
    def forward(self, x): 
        	return self.fc(x)


umap_configs = [
    {'n_components': rfs, 'n_neighbors': nn}#, 'random_state': 42}
    for rfs in config['umap__reduced_fe_size']
    for nn in config['umap__neighbors']
]

pca_configs = [
    {'n_components': rfs, 'random_state': 42}
    for rfs in config['pca__reduced_fe_size']
]

fe_reduction_configs = {
    'PCA': pca_configs,
    'UMAP': umap_configs
}


for reducer_name, param_list in fe_reduction_configs.items():
    for params in param_list:
        if reducer_name == 'PCA':
            reducer = PCA(**params)
        elif reducer_name == 'UMAP':
            reducer = UMAP(**params)
        
        print(f"Applying {reducer_name}...")
        start_time = time.time()
        X_train_reduced = reducer.fit_transform(X_train)
        X_temp_reduced = reducer.transform(X_temp) if hasattr(reducer, 'transform') else reducer.fit_transform(X_temp)
        X_val_reduced = reducer.transform(X_val) if hasattr(reducer, 'transform') else reducer.fit_transform(X_val)
        X_test_reduced = reducer.transform(X_test) if hasattr(reducer, 'transform') else reducer.fit_transform(X_test)
        reduction_time = time.time() - start_time

        print(f'SHAPES: normal train: {X_train.shape} | reduced train: {X_train_reduced.shape}')
        print(f'SHAPES: normal test: {X_test.shape} | reduced test: {X_test_reduced.shape}')

        for model_name in model_names:

            if model_name == "Linear Classifier":
                for epochs in config['linear__epochs']:
                    for lr in config['linear__learning_rate']:

                        print(f"Training {model_name} with {reducer_name} feature embeddings [LR: {lr} | Epochs: {epochs}]")
                        val_accuracies, val_losses = [], []
                        # PyTorch Model Setup
                        input_dim = X_train_reduced.shape[1]
                        num_classes = 277

                        start_time = time.time()

                        linear_model = LinearClassifier(input_dim, num_classes).to(device)
                        criterion = nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(linear_model.parameters(), lr=lr)
                        #scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
                        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=pat, min_lr=0.0001)

                        X_train_tensor = torch.tensor(X_train_reduced, dtype=torch.float32).to(device)
                        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)  # Convert to Long type for CrossEntropyLoss
                        X_val_tensor = torch.tensor(X_val_reduced, dtype=torch.float32).to(device)
                        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

                        for epoch in range(epochs):
                            linear_model.train()
                            optimizer.zero_grad()

                            outputs = linear_model(X_train_tensor)
                            loss = criterion(outputs, y_train_tensor)
                            loss.backward()
                            optimizer.step()
                            #scheduler.step(loss)

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
                        X_test_tensor = torch.tensor(X_test_reduced, dtype=torch.float32).to(device)
                        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)  # Convert to Long type for evaluation
                        with torch.no_grad():
                            outputs = linear_model(X_test_tensor)
                            _, y_pred = torch.max(outputs, 1)  # Get the predicted class indices
                            y_pred_numpy = y_pred.cpu().numpy()

                        training_time = time.time() - start_time

                        handle_results(csv_file_path, y_test, y_pred_numpy, reducer_name, model_name, 
                                    params, reduction_time, training_time, neighbors=None,
                                    lr=lr, epochs=epochs, val_accuracies=val_accuracies, val_losses=val_losses)

            elif model_name == 'KNN':
                for neighbors in config['knn__neighbors']:
                    print(f"Training {model_name} with {reducer_name} feature embeddings [K: {neighbors}]")
                    start_time = time.time()

                    model = KNeighborsClassifier(n_neighbors=neighbors)
                    model.fit(X_train_reduced, y_train)
                    y_pred_numpy = model.predict(X_temp_reduced)
                    training_time = time.time() - start_time

                    handle_results(csv_file_path, y_temp, y_pred_numpy, reducer_name, model_name, 
                                params, reduction_time, training_time, neighbors=neighbors,
                                lr=None, epochs=None, accuracies=None, losses=None, C=None)
            else:
                print(f'INVALID MODEL NAME: {model_name}')
                sys.exit(1)


