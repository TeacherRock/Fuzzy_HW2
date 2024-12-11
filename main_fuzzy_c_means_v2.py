import pandas as pd
from itertools import combinations
from fcmeans import FCM
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

def fuzzy_c_means_clustering(X):
    fcm = FCM(n_clusters=2, random_state=42)
    fcm.fit(X.to_numpy())
    return fcm.centers, fcm.u.argmax(axis=1) + 1

def evaluate_features(X, y_true):
    centers, cluster_labels = fuzzy_c_means_clustering(X)
    accuracy = accuracy_score(y_true, cluster_labels)
    ari = adjusted_rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)
    return accuracy, ari, nmi

def find_best_features(df, y_true, max_features=9):
    feature_names = df.columns
    best_score = 0
    best_features = None
    best_metrics = None

    for k in range(1, max_features + 1):
        for subset in combinations(feature_names, k):
            X_subset = df[list(subset)]
            accuracy, ari, nmi = evaluate_features(X_subset, y_true)
            score = ari  # Choose ARI or NMI as the primary metric for selection

            if score > best_score:
                best_score = score
                best_features = subset
                best_metrics = (accuracy, ari, nmi)

    return best_features, best_metrics

def main():
    # Load dataset
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    # Drop the 'Classification' column to use only features for clustering
    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]

    # Find the best feature subset
    best_features, best_metrics = find_best_features(X, y_true)

    print("\nBest Features:", best_features)
    print(f"Best Metrics: Accuracy={best_metrics[0]:.4f}, ARI={best_metrics[1]:.4f}, NMI={best_metrics[2]:.4f}")

if __name__ == "__main__":
    main()
