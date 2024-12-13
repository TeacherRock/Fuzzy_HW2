import pandas as pd
from itertools import combinations
from fcmeans import FCM
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def fuzzy_c_means_clustering(X, m=2):
    fcm = FCM(n_clusters=2, m=m, random_state=42)
    fcm.fit(X.to_numpy())
    return fcm.centers, fcm.u.argmax(axis=1) + 1

def evaluate_features(X, y_true, m=2):
    centers, cluster_labels = fuzzy_c_means_clustering(X, m=m)
    accuracy = accuracy_score(y_true, cluster_labels)
    return accuracy, cluster_labels

def find_best_features(df, y_true, max_features=9, find_feature=False):
    feature_names = df.columns
    best_score = 0
    best_features = None
    best_metrics = None
    best_predictions = None
    best_m = None
    for m in range(2, 20):
        if find_feature:
            for k in range(1, max_features + 1):
                for subset in combinations(feature_names, k):
                    X_subset = df[list(subset)]
                    accuracy, cluster_labels = evaluate_features(X_subset, y_true, m)
                    score = accuracy

                    if score > best_score:
                        best_score = score
                        best_features = subset
                        best_metrics = accuracy
                        best_predictions = cluster_labels
                        best_m = m
        else:
            X_subset = df[list(feature_names)]
            accuracy, cluster_labels = evaluate_features(X_subset, y_true, m)
            score = accuracy

            if score > best_score:
                best_score = score
                best_metrics = accuracy
                best_predictions = cluster_labels
                best_m = m

    return best_features, best_metrics, best_predictions, best_m

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

    best_features, best_metrics, best_predictions, best_m = find_best_features(X_normalized, y_true, find_feature=find_feature)
    feature = "_best" if find_feature else "_all"
    save_results_to_txt(best_predictions, y_true, "./result/fuzzy_c_means_norm" + feature + "_results.txt")

    print("Best Features:", best_features)
    print("Best m:", best_m)
    print(f"Best Metrics: Accuracy={best_metrics:.4f}")

def main_wo_normalize(find_feature=False):
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]

    best_features, best_metrics, best_predictions, best_m = find_best_features(X, y_true, find_feature=find_feature)
    feature = "_best" if find_feature else "_all"
    save_results_to_txt(best_predictions, y_true, "./result/fuzzy_c_means" + feature + "_results.txt")

    print("Best Features:", best_features)
    print("Best m:", best_m)
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
