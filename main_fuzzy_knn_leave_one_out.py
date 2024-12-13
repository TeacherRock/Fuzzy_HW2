import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

def fuzzy_k_nearest_neighbor_clustering(X_train, y_train, X_test, k=3, m=2):
    def calculate_membership_matrix(X, y_train, X_test, k, m):
        membership_matrix = np.zeros((len(X_test), len(np.unique(y_train))))

        for i, test_sample in enumerate(X_test):
            distances = np.linalg.norm(X - test_sample, axis=1)
            sorted_indices = np.argsort(distances)[:k]

            for j in sorted_indices:
                for label in np.unique(y_train):
                    if y_train[j] == label:
                        membership_matrix[i, label - 1] += 1 / (distances[j] ** (2 / (m - 1)))

            membership_matrix[i] /= membership_matrix[i].sum()
        return membership_matrix

    membership_matrix = calculate_membership_matrix(X_train, y_train, X_test, k, m)
    cluster_labels = membership_matrix.argmax(axis=1) + 1
    return cluster_labels

def leave_one_out_evaluation(X, y_true, k=3, m=2):
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    accuracies = []
    all_predictions = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
        cluster_label = fuzzy_k_nearest_neighbor_clustering(X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), k, m)
        accuracy = accuracy_score(y_test, cluster_label)
        accuracies.append(accuracy)
        all_predictions.append((test_index[0], cluster_label[0]))

    avg_accuracy = sum(accuracies) / len(accuracies)
    return avg_accuracy

def find_best_features_leave_one_out(df, y_true, max_features=9, best_feature=[], m=2):
    feature_names = best_feature

    X_subset = df[list(feature_names)]
    avg_accuracy = leave_one_out_evaluation(X_subset, y_true)

    return avg_accuracy

def main_wo_normalize(best_feature=[]):
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]

    avg_accuracy = find_best_features_leave_one_out(X, y_true, best_feature=best_feature)
    print(f"Best Average Accuracy (Leave-One-Out): {avg_accuracy:.4f}")

def main_normalize(best_feature=[]):
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]
    scaler = StandardScaler()  # You can use MinMaxScaler() for scaling between 0 and 1
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    avg_accuracy = find_best_features_leave_one_out(X_normalized, y_true, best_feature=best_feature)
    print(f"Best Average Accuracy (Leave-One-Out): {avg_accuracy:.4f}")

if __name__ == "__main__":
    
    all_feature = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']

    best_feature = ['Age', 'Glucose', 'Resistin']
    ## Normalize ##
    print("=============== Normalize, Use best featrue =================")
    main_normalize(best_feature=best_feature) ## Best featrue

    print("=============== Normalize, Use all featrue ==================")
    main_normalize(best_feature=all_feature) ## Best featrue
    
    best_feature = ['Age', 'Glucose', 'Resistin']
    ## not Normalize ##
    print("=============== not Normalize, Use best featrue =============")
    main_wo_normalize(best_feature=best_feature) ## Best featrue

    print("=============== Normalize, Use all featrue ==================")
    main_normalize(best_feature=all_feature) ## Best featrue
