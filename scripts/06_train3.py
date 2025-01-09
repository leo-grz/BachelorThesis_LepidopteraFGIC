import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import check_folder_exists, load_features, save_features

PATH_TO_DATA = "/home/lgierz/BA_MothClassification/data/"
feature_file = PATH_TO_DATA + f'processed/cv_datasets/resnet_feature_dataset_top277_max3000.npz' 



features, labels, _ = load_features(feature_file)

print(features.shape)
print(labels.shape)

# TODO: during datasplit maintain class balance
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)




import numpy as np
from sklearn.decomposition import PCA
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# # Example Data
# np.random.seed(42)
# X = np.random.rand(200000, 2048)  # 200k samples, 2048 features
# y = np.random.randint(0, 10, 200000)  # 10 classes

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dimensionality Reduction with PCA
components = [128, 256]
k = [25, 30, 35, 50, 60]

for c in components:

    print(f'Loading PCA for {c} components')
    pca = PCA(n_components=c)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

#print('Loading UMAP')
# Dimensionality Reduction with UMAP
#umap_reducer = umap.UMAP(n_neighbors=15, n_components=128, random_state=42, n_jobs=24)
#X_train_umap = umap_reducer.fit_transform(X_train)
#X_test_umap = umap_reducer.transform(X_test)

#print('Training PCA KNN')
# Train and Evaluate KNN for PCA
    for num in k:
        knn_pca = KNeighborsClassifier(n_neighbors=num)
        knn_pca.fit(X_train_pca, y_train)
        y_pred_pca = knn_pca.predict(X_test_pca)
        accuracy_pca = accuracy_score(y_test, y_pred_pca)
        print(f"PCA + KNN Accuracy, {c} components and {num} for k: {accuracy_pca:.4f}")
#print('Training UMAP KNN')
# Train and Evaluate KNN for UMAP
#knn_umap = KNeighborsClassifier(n_neighbors=15)
#knn_umap.fit(X_train_umap, y_train)
#y_pred_umap = knn_umap.predict(X_test_umap)
#accuracy_umap = accuracy_score(y_test, y_pred_umap)
#print(f"UMAP + KNN Accuracy: {accuracy_umap:.4f}")









param_grid = {
    'n_neighbors': [9,14,19],  # Number of neighbors to use
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
    'leaf_size': [20, 30, 40],  # Leaf size passed to BallTree or KDTree
    'p': [1, 2]  # Power parameter for the Minkowski metric
}

# TODO: Implement Stratified k-Fold Cross validation for Training dataset
# TODO: macht multithreading sinn?

# for k in param_grid['n_neighbors']:
#     for w in param_grid['weights']:
#         for p in param_grid['p']:


#             print(f'---------------------- K = {k} | Weights = {w} | P = {p}')
#             knn = KNeighborsClassifier(n_neighbors=k, weights=w, p=p)


#             knn.fit(X_train, y_train)

#             y_pred = knn.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             print(f"Accuracy: {accuracy * 100:.2f}%\n")



