from knn import load_data, load_knn_data, save_knn_data, advanced_metrics#
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import cv2
import os
import kagglehub
from math import *
from tqdm import tqdm
import joblib

def save_svm_model(model, filename):
    joblib.dump(model, filename)
    print(f"Modèle SVM sauvegardé sous : {filename}")
 
def load_svm_model(filename):
    if os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Modèle SVM chargé depuis : {filename}")
        return model
    return None

def train_svm(X_train, y_train, kernel="rbf", C=1.0):
    print(f"-" * 20 + "Entrainement du SVM avec Kernel : {kernel}" + "-" * 20)
    model = SVC(kernel=kernel, C=C, class_weight="balanced")
    model.fit(X_train, y_train)
    save_svm_model(model, "medical_svm")
    print(f"-"*20 + "Entrainement du SVM terminé" + "-"*20)
    return model

def evaluate_svm(model, X_test, y_test):
    print(f"-" * 20 + "Evaluation du SVM" + "-" * 20)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    C = confusion_matrix(y_test, y_pred)
    
    print(f"-" * 20 + "Evaluation du SVM terminée" + "-" * 20)
    
    C = C.T 
    
    return accuracy, C

if __name__ == "__main__":

    # class_names = ["notumor", "glioma"]
    class_names = ["no_tumor", "glioma_tumor"] # Dataset 2

    # data_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset") # Dataset
    data_path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri") # Dataset 2
    print("Path to dataset files:", data_path)

    print("\n" + "-"*60)
    print(f" SVM Classifier ".center(60))
    print("=" * 60)

    # data = load_knn_data("knn_cache.npz")
    data = load_knn_data("knn_cache_dataset_2.npz")
    if data is None:
        X_train, y_train = load_data(data_path, class_names, train_step="Training")
        X_test, y_test = load_data(data_path, class_names, train_step="Testing")
        save_knn_data(X_train, y_train, X_test, y_test)
    else:
        X_train, y_train, X_test, y_test = data

    svm_model = load_svm_model("medical_svm")
    if svm_model is None:
        svm_model = train_svm(X_train, y_train)

    accuracy, C = evaluate_svm(svm_model, X_test, y_test)
    recall, precision, f1_score = advanced_metrics(C)
    correct_samples = np.trace(C)
    total_samples = np.sum(C)

    print("="*60)
    print("  RESULTS SUMMARY ".center(60))
    print("="*60)
    print(f"| Accuracy : {accuracy:.4f} ({accuracy*100:.2f} %)".ljust(59) + "|")
    print(f"| Recall : {recall:.4f} ({recall*100:.2f} %)".ljust(59) + "|")
    print(f"| Precision : {precision:.4f} ({precision*100:.2f} %)".ljust(59) + "|")
    print(f"| F1-Score : {f1_score:.4f} ({f1_score*100:.2f} %)".ljust(59) + "|")
    print(f"| Correct Predictions : {correct_samples}/{total_samples}".ljust(59) + "|")
    print("="*60)
    print(f"-------------------------- END ---------------------------")



