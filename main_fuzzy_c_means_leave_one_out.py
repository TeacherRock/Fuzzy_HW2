
import numpy as np
import pandas as pd
from itertools import combinations
from fcmeans import FCM
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

def fuzzy_c_means_clustering(X, m=2):
    fcm = FCM(n_clusters=2, m=m, random_state=42)
    fcm.fit(X.to_numpy())
    return fcm.centers, fcm.u.argmax(axis=1) + 1

def assign_to_nearest_cluster(X_test, centers):
    distances = np.linalg.norm(centers - X_test.to_numpy(), axis=1)
    return distances.argmin() + 1


def leave_one_out_evaluation(X, y_true):
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    accuracies = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_test = y_true.iloc[test_index]

        centers, cluster_labels = fuzzy_c_means_clustering(X_train, m=m)

        test_label = assign_to_nearest_cluster(X_test, centers)

        accuracy = accuracy_score(y_test, [test_label])
        accuracies.append(accuracy)
    avg_accuracy = sum(accuracies) / len(accuracies)

    return avg_accuracy

def best_features_leave_one_out(df, y_true, max_features=9, best_feature=[], m=2):
    feature_names = best_feature

    X_subset = df[list(feature_names)]
    avg_accuracy = leave_one_out_evaluation(X_subset, y_true)
    return avg_accuracy

def find_best_features_leave_one_out(df, y_true, max_features=9, m=2):
    feature_names = df.columns
    best_score = 0
    best_features = None
    best_metrics = None
    for k in range(1, max_features + 1):
        for subset in combinations(feature_names, k):
            X_subset = df[list(subset)]
            accuracy = leave_one_out_evaluation(X_subset, y_true)
            score = accuracy

            if score > best_score:
                best_score = score
                best_features = subset
                best_metrics = accuracy

    return best_features, best_metrics

def main_normalize(best_feature=[], find_best_featrue=False, m=2):
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]

    scaler = StandardScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if find_best_featrue:
        best_features, avg_accuracy = find_best_features_leave_one_out(X, y_true)
        print("Best Features:", best_features)
        print(f"Best Average Accuracy (Leave-One-Out): {avg_accuracy:.4f}")
    else:
        avg_accuracy = best_features_leave_one_out(X, y_true, best_feature=best_feature)
        print(f"Best Average Accuracy (Leave-One-Out): {avg_accuracy:.4f}")

def main_wo_normalize(best_feature=[], find_best_featrue=False, m=2):
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]

    if find_best_featrue:
        best_features, avg_accuracy = find_best_features_leave_one_out(X, y_true)
        print("Best Features:", best_features)
        print(f"Best Average Accuracy (Leave-One-Out): {avg_accuracy:.4f}")
    else:
        avg_accuracy = best_features_leave_one_out(X, y_true, best_feature=best_feature)
        print(f"Best Average Accuracy (Leave-One-Out): {avg_accuracy:.4f}")

if __name__ == "__main__":
    
    all_feature = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']

    best_feature = ['Glucose', 'HOMA', 'Resistin']
    m = 11
    ## Normalize ##
    print("=============== Normalize, Use best featrue =================")
    print("Best Features:", best_feature)
    main_normalize(best_feature=best_feature, m=m) ## Best featrue

    m = 8
    print("=============== Normalize, Use all featrue ==================")
    main_normalize(best_feature=all_feature, m=m) ## Best featrue
    
    best_feature = ['BMI', 'Glucose', 'Adiponectin', 'Resistin']
    m = 13
    ## not Normalize ##
    print("Best Features:", best_feature)
    print("=============== not Normalize, Use best featrue =============")
    main_wo_normalize(best_feature=best_feature, m=m) ## Best featrue

    m = 2
    print("=============== Normalize, Use all featrue ==================")
    main_normalize(best_feature=all_feature, m=m) ## Best featrue

    print("\n================= Find new best feature =====================")
    print("=============== not Normalize=============")
    main_wo_normalize(find_best_featrue=True)

    print("=============== Normalize==================")
    main_normalize(find_best_featrue=True)
