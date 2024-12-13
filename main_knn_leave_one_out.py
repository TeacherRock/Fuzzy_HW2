import pandas as pd
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

def nearest_neighbor_clustering(X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    cluster_label = knn.predict(X_test)
    return cluster_label

def leave_one_out_evaluation(X, y_true):
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    accuracies = []
    all_predictions = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
        cluster_label = nearest_neighbor_clustering(X_train, y_train, X_test)
        accuracy = accuracy_score(y_test, cluster_label)
        accuracies.append(accuracy)
        all_predictions.append((test_index[0], cluster_label[0]))

    avg_accuracy = sum(accuracies) / len(accuracies)
    return avg_accuracy

def find_best_features_leave_one_out(df, y_true, max_features=9, best_feature=[]):
    feature_names = best_feature

    X_subset = df[list(feature_names)]
    avg_accuracy = leave_one_out_evaluation(X_subset, y_true)

    return avg_accuracy

def main_normalize(best_feature=[]):
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]

    scaler = StandardScaler()  # You can use MinMaxScaler() for scaling between 0 and 1
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    best_score = find_best_features_leave_one_out(X_normalized, y_true, best_feature=best_feature)
    print(f"Best Average Accuracy (Leave-One-Out):  Accuracy={best_score:.4f}")

def main_wo_normalize(best_feature=[]):
    data_path = "./data/dataR2.csv"
    df = pd.read_csv(data_path)

    X = df.drop("Classification", axis=1)
    y_true = df["Classification"]

    best_score = find_best_features_leave_one_out(X, y_true, best_feature=best_feature)
    print(f"Best Average Accuracy (Leave-One-Out):  Accuracy={best_score:.4f}")


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
