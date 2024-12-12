import pandas as pd
from itertools import combinations
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import numpy as np

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
    ari = adjusted_rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)
    return accuracy, ari, nmi, cluster_labels

def find_best_features(df, y_true, max_features=9):
    feature_names = df.columns
    best_score = 0
    best_features = None
    best_metrics = None
    best_predictions = None

    for k in range(1, max_features + 1):
        for subset in combinations(feature_names, k):
            X_subset = df[list(subset)]
            accuracy, ari, nmi, cluster_labels = evaluate_features(X_subset, y_true)
            score = accuracy

            if score > best_score:
                best_score = score
                best_features = subset
                best_metrics = (accuracy, ari, nmi)
                best_predictions = cluster_labels

    return best_features, best_metrics, best_predictions

def save_results_to_txt(predictions, ground_truth, filename):
    results = pd.DataFrame({
        "pred": predictions,
        "answer": ground_truth
    })
    results.to_csv(filename, sep="\t", index=False)
    print(f"Results saved to {filename}.")

def main():
    # Load dataset
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    # Drop the 'Classification' column to use only features for clustering
    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]

    # Find the best feature subset
    best_features, best_metrics, best_predictions = find_best_features(X, y_true)
    save_results_to_txt(best_predictions, y_true, "./result/fuzzy_knn_best_results.txt")

    print("\nBest Features:", best_features)
    print(f"Best Metrics: Accuracy={best_metrics[0]:.4f}, ARI={best_metrics[1]:.4f}, NMI={best_metrics[2]:.4f}")

if __name__ == "__main__":
    main()
