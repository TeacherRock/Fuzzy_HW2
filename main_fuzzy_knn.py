import pandas as pd
from itertools import combinations
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler

def fuzzy_k_nearest_neighbor_clustering(X, y_true, k=2, m=2):
    def calculate_membership_matrix(X, k, m):
        n_samples = X.shape[0]
        membership_matrix = np.zeros((n_samples, len(np.unique(y_true))))
        
        for i in range(n_samples):
            distances = np.linalg.norm(X - X[i], axis=1)
            distances[i] = np.inf
            sorted_indices = np.argsort(distances)[:k]

            for j in sorted_indices:
                for label in np.unique(y_true):
                    if y_true[j] == label:
                        membership_matrix[i, label - 1] += 1 / (distances[j] ** (2 / (m - 1)))
            
            membership_matrix[i] /= membership_matrix[i].sum()
        return membership_matrix

    membership_matrix = calculate_membership_matrix(X.to_numpy(), k, m)
    cluster_labels = membership_matrix.argmax(axis=1) + 1
    return cluster_labels

def evaluate_features(X, y_true):
    cluster_labels = fuzzy_k_nearest_neighbor_clustering(X, y_true)
    accuracy = accuracy_score(y_true, cluster_labels)
    return accuracy, cluster_labels

def find_best_features(df, y_true, max_features=9, find_feature=False):
    feature_names = df.columns
    best_score = 0
    best_features = None
    best_metrics = None
    best_predictions = None
    if find_feature:
        for k in range(1, max_features + 1):
            for subset in combinations(feature_names, k):
                X_subset = df[list(subset)]
                accuracy, cluster_labels = evaluate_features(X_subset, y_true)
                score = accuracy

                if score > best_score:
                    best_score = score
                    best_features = subset
                    best_metrics = accuracy
                    best_predictions = cluster_labels
    else:
        X_subset = df[list(feature_names)]
        accuracy, cluster_labels = evaluate_features(X_subset, y_true)
        score = accuracy
        if score > best_score:
            best_score = score
            best_metrics = accuracy
            best_predictions = cluster_labels

    return best_features, best_metrics, best_predictions

def save_results_to_txt(predictions, ground_truth, filename):
    results = pd.DataFrame({
        "pred": predictions,
        "answer": ground_truth
    })
    results.to_csv(filename, sep="\t", index=False)
    print(f"Results saved to {filename}.")

def main_normalize(find_feature=False):
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]

    scaler = StandardScaler()  # You can use MinMaxScaler() for scaling between 0 and 1
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    best_features, best_metrics, best_predictions = find_best_features(X, y_true, find_feature=find_feature)
    feature = "_best" if find_feature else "_all"
    save_results_to_txt(best_predictions, y_true, "./result/nearest_neighbor" + feature + "_results.txt")

    print("Best Features:", best_features)
    print(f"Best Metrics: Accuracy={best_metrics:.4f}")

def main_wo_normalize(find_feature=False):
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]

    best_features, best_metrics, best_predictions = find_best_features(X, y_true, find_feature=find_feature)
    feature = "_best" if find_feature else "_all"
    save_results_to_txt(best_predictions, y_true, "./result/nearest_neighbor" + feature + "_results.txt")

    print("Best Features:", best_features)
    print(f"Best Metrics: Accuracy={best_metrics:.4f}")

if __name__ == "__main__":
    ## Normalize ##
    print("=============== Normalize, Find best featrue =================")
    main_normalize(find_feature=True) ## Best featrue
    print("=============== Normalize, Use all featrue ===================")
    main_normalize(find_feature=False) ## All featrue

    ## not Normalize ##
    print("=============== not Normalize, Find best featrue =============")
    main_wo_normalize(find_feature=True) ## Best featrue
    print("=============== not Normalize, Use all featrue ===============")
    main_wo_normalize(find_feature=False) ## All featrue
