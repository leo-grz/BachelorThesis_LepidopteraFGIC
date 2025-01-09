import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # SVM Import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os
import random
import logging
import torch
from torch import nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import check_folder_exists, load_features, save_features

PATH_TO_DATA = '/home/lgierz/BA_MothClassification/data/'
PATH_TO_FEATURES = PATH_TO_DATA + 'processed/features/'
PATH_TO_LOGFILE = PATH_TO_DATA + 'status/06_train4.log'
feature_file = PATH_TO_DATA + 'processed/cv_datasets/resnet_feature_dataset_top277_max3000.npz' 

# Configure logging
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)

components = [64, 128, 256, 512, 1024]

# Results Storage
results = []

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

print(np.min(y_train))
print(np.max(y_train))
print(len(np.unique(y_train)))
print(len(np.unique(y_test)))

# Assuming y_train and y_test are your original labels
unique_labels = np.unique(y_train)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# Map the original labels to new continuous indices
y_train = np.array([label_mapping[label] for label in y_train])
y_test = np.array([label_mapping[label] for label in y_test])

# # Verify the new labels
# print("Original Labels:", np.unique(y_train))
# print("Mapped Labels:", np.unique(y_train_mapped))
# print("Number of different numbers in y_train_mapped:", len(np.unique(y_train_mapped)))



# X_train = X_train.to(device)
# X_test = X_test.to(device)
# y_train = y_train.to(device)
# y_test = y_test.to(device)
# Linear Classifier Model in PyTorch
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes): 
        super(LinearClassifier, self).__init__() 
        ##self.fc = nn.Linear(input_dim, num_classes) 
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(512, num_classes) # try also 1024, 256, 128?
        )
        
    def forward(self, x): 
        	return self.fc(x)
# GPU-Unterstützung (falls verfügbar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for num in components:
    methods = {
        "PCA": PCA(n_components=num),
        #"t-SNE": TSNE(n_components=num, random_state=42),
        "UMAP": UMAP(n_components=num, random_state=42),
        "LDA": LinearDiscriminantAnalysis(n_components=num),
        "Kernel PCA": KernelPCA(n_components=num, kernel='rbf')
    }
    print(f'Initializing methods with {num} n_components...')

    for method_name, reducer in methods.items():
        print(f"Applying {method_name}...")
        start_time = time.time()
        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test) if hasattr(reducer, 'transform') else reducer.fit_transform(X_test)
        reduction_time = time.time() - start_time

        print(f'SHAPES: normal train: {X_train.shape} | reduced train: {X_train_reduced.shape}')
        print(f'SHAPES: normal test: {X_test.shape} | reduced test: {X_test_reduced.shape}')

        # Models to Train
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=25),
            "Linear Classifier": None  # Platzhalter für PyTorch-Modell
        }

        for model_name, model in models.items():
            start_time = time.time()

            if model_name == "Linear Classifier":
                print(f"Training {model_name} with {method_name} features...")

                # PyTorch Model Setup
                input_dim = X_train_reduced.shape[1]
                num_classes = 277
                linear_model = LinearClassifier(input_dim, num_classes).to(device)
                criterion = nn.CrossEntropyLoss()
                #optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.01)
                optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.001)
                # Training Loop
                epochs = 50
                X_train_tensor = torch.tensor(X_train_reduced, dtype=torch.float32).to(device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)  # Convert to Long type for CrossEntropyLoss

                for epoch in range(epochs):
                    linear_model.train()
                    optimizer.zero_grad()

                    outputs = linear_model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()

                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

                # Evaluate Linear Classifier
                linear_model.eval()
                X_test_tensor = torch.tensor(X_test_reduced, dtype=torch.float32).to(device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)  # Convert to Long type for evaluation
                with torch.no_grad():
                    outputs = linear_model(X_test_tensor)
                    _, y_pred = torch.max(outputs, 1)  # Get the predicted class indices
                    y_pred_numpy = y_pred.cpu().numpy()

            else:
                print(f"Training {model_name} with {method_name} features...")
                model.fit(X_train_reduced, y_train)
                y_pred_numpy = model.predict(X_test_reduced)
            training_time = time.time() - start_time

            # Evaluation Metrics
            #y_pred_numpy = y_pred.cpu().numpy()
            acc = accuracy_score(y_test, y_pred_numpy)
            prec = precision_score(y_test, y_pred_numpy, average='weighted')
            rec = recall_score(y_test, y_pred_numpy, average='weighted')
            f1 = f1_score(y_test, y_pred_numpy, average='weighted')

            # Store Results
            run_results = {
                "Method": method_name,
                "Model": model_name,
                "Components": num,
                "Reduction Time (s)": reduction_time,
                "Training Time (s)": training_time,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1
            }
            logging.info(f"[{run_results['Method']} ({run_results['Components']})][{run_results['Model']}] Reduction time: {run_results['Reduction Time (s)']:.2f}s, Training time: {run_results['Training Time (s)']:.2f}s")
            logging.info(f"[{run_results['Method']} ({run_results['Components']})][{run_results['Model']}] Accuracy: {run_results['Accuracy']:.2f}, Precision: {run_results['Precision']:.2f}, Recall: {run_results['Recall']:.2f}, F1-Score: {run_results['F1-Score']:.2f}")

            results.append(run_results)

# Print Results
for result in results:
    print(result)
