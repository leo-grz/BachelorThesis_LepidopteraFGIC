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
import joblib
from torch import nn
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils_helpers import check_folder_exists, load_features, save_features


PATH_TO_DATA = '/home/lgierz/BA_MothClassification/data/'
PATH_TO_CA = PATH_TO_DATA + 'confidance_analysis/'
PATH_TO_DATASETS = PATH_TO_DATA + 'processed/cv_datasets/'
PATH_TO_LOGFILE = PATH_TO_DATA + 'status/confidance_tests1.log'
csv_file_path = PATH_TO_DATA + 'status/confidance_tests1.csv'

model_names = ["Linear Classifier", "KNN"]
fm_names = ['ResNet', 'DINO']


config = {    
    'pca__reduced_fe_size': 512,

    'knn__neighbors': 35,

    'linear__learning_rate': 0.001,
    'linear__epochs': 1500,
    'linear__patience': 3,
    'linear__gamma': 0.8
}

dataset_configs = {
    'top277': (277, [3000, 2000, 1000, 500]),
    'top387': (387, [2000, 1000, 500]),
    'top589': (589, [1000, 500])
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


def handle_results(csv_path, test, pred, species, samples, fm, model, training_time, neighbors=None, lr=None, epochs=None, losses=None, accuracies=None, gamma=None, patience=None, confidences=None, gbifids=None):
    acc = accuracy_score(test, pred)
    prec = precision_score(test, pred, average='weighted')
    rec = recall_score(test, pred, average='weighted')
    f1 = f1_score(test, pred, average='weighted')

    # Store Results
    run_results = {
        "SpeciesAmount": species,
        "SampleAmount": samples,
        "FoundationalModel": fm,
        "Model": model,
        "Training Time (s)": round(training_time, 2),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Neighbors": neighbors if model == 'KNN' else None,
        "Learning Rate": lr if model == "Linear Classifier" else None,
        "Epochs": epochs if model == "Linear Classifier" else None,
        "Gamma": gamma if model == "Linear Classifier" else None,
        "Patience": patience if model == "Linear Classifier" else None,
        "Epoch Losses": losses if model == "Linear Classifier" else None,
        "Epoch Accuracies": accuracies if model == "Linear Classifier" else None,
    }
    # Convert the dictionary to a DataFrame
    results_df = pd.DataFrame([run_results])  # Wrap in a list to create a DataFrame with one row
    if not os.path.isfile(csv_path):
        results_df.to_csv(csv_path, index=False)
    else:
        results_df.to_csv(csv_path, mode='a', header=False, index=False)

    logging.info(f"[{run_results['SpeciesAmount']} Species][{run_results['SampleAmount']} Samples][{run_results['FoundationalModel']}][{run_results['Model']}] Training time: {run_results['Training Time (s)']}s, Accuracy: {run_results['Accuracy']}, Precision: {run_results['Precision']}, Recall: {run_results['Recall']}, F1-Score: {run_results['F1-Score']}")

    if confidences is not None and gbifids is not None:
        np.savez(PATH_TO_CA + f"predictions/predictions_max{species}top{samples}_{fm}_{model}.npz",gbifids=gbifids, labels=test, predicted_labels=pred, confidances=confidences)
        print(f"predictions_max{species}top{samples}_{fm}_{model}.npz saved")


for fm in fm_names:
    fm = fm.lower()

    # Iterate through dataset configurations
    for dataset_name, (class_amount, sample_amounts) in dataset_configs.items():
        
        # Iterate through the sample amounts
        for sample_amount in sample_amounts:
            print(f'Processing {fm} dataset with {class_amount} classes and {sample_amount} samples')
            feature_file_train = PATH_TO_CA + 'split_datasets/' + f'{fm}_feature_dataset_top{class_amount}_max{sample_amount}_train.npz'
            feature_file_test = PATH_TO_CA + 'split_datasets/' + f'{fm}_feature_dataset_top{class_amount}_max{sample_amount}_test.npz'

            X_train, y_train, gbifids_train = load_features(feature_file_train)
            X_test, y_test, gbifids_test = load_features(feature_file_test)

            print(f'Lowest label: {np.min(y_train)}, highest label: {np.max(y_train)}, unique labels in training ds: {len(np.unique(y_train))}, unique labels in testing ds: {len(np.unique(y_test))}')
            print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}, gbifids_train shape: {gbifids_train.shape}, gbifids_test shape: {gbifids_test.shape}')

            # fix missing labels due to dataset splitting by changing label to ascending order
            label_mapping = {label: idx for idx, label in enumerate(np.unique(y_train))}
            y_train = np.array([label_mapping[label] for label in y_train]) # since labels are not in ascending order, remapping is necessary
            y_test = np.array([label_mapping[label] for label in y_test])

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


            class LinearClassifier(nn.Module):
                def __init__(self, input_dim, num_classes): 
                    super(LinearClassifier, self).__init__() 
                    self.fc = nn.Linear(input_dim, num_classes) 
                    
                def forward(self, x): 
                        return self.fc(x)
                
            reducer = PCA(n_components=config['pca__reduced_fe_size'])

            X_train_reduced = reducer.fit_transform(X_train)
            X_test_reduced = reducer.transform(X_test) if hasattr(reducer, 'transform') else reducer.fit_transform(X_test)

            print(f'SHAPES: normal train: {X_train.shape} | reduced train: {X_train_reduced.shape}')
            print(f'SHAPES: normal test: {X_test.shape} | reduced test: {X_test_reduced.shape}')


            for model_name in model_names:

                if model_name == "Linear Classifier":
                    
                    losses, accuracies = [], []
                    # PyTorch Model Setup
                    input_dim = X_train_reduced.shape[1]
                    print('Input Dimension: ', input_dim)

                    start_time = time.time()

                    linear_model = LinearClassifier(input_dim, class_amount).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(linear_model.parameters(), lr=config['linear__learning_rate'])
                    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['linear__gamma'], patience=config['linear__patience'], min_lr=0.0001)

                    X_train_tensor = torch.tensor(X_train_reduced, dtype=torch.float32).to(device)
                    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)  # Convert to Long type for CrossEntropyLoss

                    for epoch in range(config['linear__epochs']):
                        linear_model.train()
                        optimizer.zero_grad()

                        outputs = linear_model(X_train_tensor)
                        loss = criterion(outputs, y_train_tensor)
                        loss.backward()
                        optimizer.step()
                        #scheduler.step(loss)
                    

                        # calculate accuracy
                        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class indices
                        correct = (predicted == y_train_tensor).sum().item()  # Count correct predictions
                        accuracy = correct / y_train_tensor.size(0)  # Calculate accuracy

                        losses.append(round(loss.item(), 4))
                        accuracies.append(round(accuracy, 4))

                    # Evaluate Linear Classifier
                    linear_model.eval()
                    X_test_tensor = torch.tensor(X_test_reduced, dtype=torch.float32).to(device)
                    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)  # Convert to Long type for evaluation
                    with torch.no_grad():
                        outputs = linear_model(X_test_tensor)
                        _, y_pred = torch.max(outputs, 1)  # Get the predicted class indices
                        y_pred_numpy = y_pred.cpu().numpy()
                        confidences = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()  # Get probability estimates

                    training_time = time.time() - start_time



                    print(f'\ny_test: {len(y_test)}, y_pred: {len(y_pred_numpy)}, confidences: {len(confidences)}')
                    print(f'y_test: {y_test.shape}, y_pred: {y_pred_numpy.shape}, confidences: {confidences.shape}')

                    # Save the linear model
                    handle_results(csv_file_path, y_test, y_pred_numpy,fm=fm, species=class_amount, samples=sample_amount, model=model_name, training_time=training_time,
                                lr=config['linear__learning_rate'], epochs=config['linear__epochs'], accuracies=accuracies, losses=losses, confidences=confidences, gbifids=gbifids_test)

                    torch.save(linear_model.state_dict(),  PATH_TO_CA + f'models/model_top{class_amount}max{sample_amount}_{fm}_{model_name}.pth')


                elif model_name == 'KNN':
                    start_time = time.time()

                    model = KNeighborsClassifier(n_neighbors=config['knn__neighbors'])
                    model.fit(X_train_reduced, y_train)
                    y_pred_numpy = model.predict(X_test_reduced)
                    confidences = model.predict_proba(X_test_reduced)  # Get probability estimates

                    training_time = time.time() - start_time

                    
                    # Save the KNN model
                    print(f'\ny_test: {len(y_test)}, y_pred: {len(y_pred_numpy)}, confidences: {len(confidences)}')
                    print(f'y_test: {y_test.shape}, y_pred: {y_pred_numpy.shape}, confidences: {confidences.shape}')


                    handle_results(csv_file_path, y_test, y_pred_numpy, fm=fm, species=class_amount, samples=sample_amount, model=model_name, training_time=training_time,
                                neighbors=config['knn__neighbors'], confidences=confidences, gbifids=gbifids_test)
                    
                    joblib.dump(model, PATH_TO_CA + f'models/model_top{class_amount}max{sample_amount}_{fm}_{model_name}.joblib')
                    


                else:
                    print(f'INVALID MODEL NAME: {model_name}')
                    sys.exit(1)