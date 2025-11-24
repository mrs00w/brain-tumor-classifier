import numpy as np
import cv2
import os
import kagglehub
from math import *

def load_data(data_dir, classes, train_step):
    m = 256
    # X : ce sont les images
    X_features = []
    label_id = {}
    # y : Ce sont les labels
    y = []
    for c in classes:
        label_id[c] = classes.index(c)

    for c in classes:
        path = os.path.join(data_dir, train_step, c)
        y = [i for i in range(len(classes))]

        if not os.path.isdir(path):
            print(f"[ATTENTION] : le répertoire {path} n'existe pas")
            return 

        for i in label_id.keys():
            for filename in os.listdir(path):
                img_path = os.path.join(path, filename)

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                X_features.append(img)
                y.append(label_id[i])

    return np.array(X_features), np.array(y)

def calculate_histogram(img, m):
    hist = np.array(3 * m)
    
    if img is None :
        return np.zeros(m * 3)
    
    height, width, _ = img.shape
    
    for row in img:
        for col in row:
            pixel = img[row, col]
            hist[pixel[0]] += 1
            hist[pixel[1] + m] += 1
            hist[pixel[2] + (2 * m)] += 1
            
    size = height * width
    
    if size > 0:
        hist /= size

    return hist

def predict_knn(k, X_train, y_train, hist_test):
    
    list_distance = [] #List of tuple
    
    for row in range(X_train[0]):
        distance = 0
        hist_train = X_train[row]
        for col in range(hist_train.shape[0]):
            distance += (hist_train[col] - hist_test[col]) ** 2;
        distance = sqrt(distance)
        list_distance((distance, y_train[row]))
        
    list_distance.sort(key = lambda x : x[0])
    
    k_nearest_neighbors = list_distance[:k]
    
    classes_occurence = {}
    
    for _, label in k_nearest_neighbors:
        classes_occurence[label] = classes_occurence
        
    predicted_class = max(classes_occurence.items())[0]
    
    return predicted_class

def evaluate_knn(dir_path, categories, K = 3):
    
    X_train, y_train = load_data(dir_path, categories, train_step="Testing")
    X_test, y_test = load_data(dir_path, categories, train_step="Training")
    
    num_classes = len(categories)
    num_test_examples = X_test.shape[0]
    
    C = np.zeros((num_classes, num_classes))
    
    for row in range(num_test_examples):
        c_ground_truth = y_test[row]
        hist_test = X_test[row]
        c_predicted = predict_knn(K, X_train, y_train, hist_test)
        C[c_predicted, c_ground_truth] += 1
        
    correct_predictions = np.trace(C)
    total_predictions = np.sum(C)
    
    accuracy = correct_predictions / total_predictions
    
    return accuracy


K = 4
class_names = ["glioma", "non_tumor"]

data_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print("Path to dataset files:", data_path)
print("\n" + "-"*60)
print(f" KNN Classifier (k = {K}) ".center(60))
print("=" * 60)

print(f"| Paramètre K : {K}".ljust(59) + "|")
print(f"| {len(class_names)} Classes : {', '.join(class_names)}".ljust(59) + "|")
print("=" * 60)

print("  RÉSUMÉ DES RÉSULTATS  ".center(60))
print("="*60)
accuracy = evaluate_knn(data_path, class_names, 4)

print(f"| Accuracy : {accuracy:.4f} ({accuracy*100:.2f} %)".ljust(59) + "|")
# print(f"| Prédictions Correctes : {correct_samples}".ljust(59) + "|")
# print(f"| Total d'échantillons : {total_samples}".ljust(59) + "|")
print("="*60)

print(f"-------------------------- FIN ---------------------------")





