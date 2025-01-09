import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import concurrent.futures




FOLDS = 10

def load_features(filename): 
    data = np.load(filename) 
    features = data['features'] 
    labels = data['labels'] 
    gbifids = data['gbifids']
    print(f"Features and labels loaded from {filename}") 
    return features, labels, gbifids

def evaluate_model(features, labels, n_neighbors, weights, metric):
    skf = StratifiedKFold(n_splits=FOLDS)
    accuracies = []

    for train_index, test_index in skf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    return np.mean(accuracies)

def grid_search(features, labels):
    n_neighbors_options = [3, 5, 7]
    weights_options = ['uniform', 'distance']
    metric_options = ['euclidean', 'manhattan']

    best_score = 0
    best_params = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_params = {
            executor.submit(evaluate_model, features, labels, n_neighbors, weights, metric): 
            (n_neighbors, weights, metric)
            for n_neighbors in n_neighbors_options
            for weights in weights_options
            for metric in metric_options
        }

        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            score = future.result()

            if score > best_score:
                best_score = score
                best_params = {
                    'n_neighbors': params[0],
                    'weights': params[1],
                    'metric': params[2]
                }
                
            print(f"Params: {params}, Score: {score}")

    print(f"Best Score: {best_score}")
    print(f"Best Params: {best_params}")

filename = 'your_data_file.npz'  # replace with your actual data file path
features, labels, gbifids = load_features(filename)
grid_search(features, labels)
