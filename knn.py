import numpy as np
import cv2
import os
import kagglehub
from math import *
from tqdm import tqdm
from skimage.feature import local_binary_pattern

def save_knn_data(X_train, y_train, X_test, y_test, filename="knn_cache_lbp_dataset_2.npz"):
    np.savez_compressed(filename, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Features KNN sauvegardées dans {filename}")

def load_knn_data(filename="knn_cache_lbp_dataset_2.npz"):
    if os.path.exists(filename):
        data = np.load(filename)
        print(f"Features KNN chargées depuis {filename}")
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']
    return None

def load_data(data_dir, classes, train_step):
    m = 256
    # X : ce sont les images
    X_features = []
    y_labels = []
    # y : Ce sont les labels
    
    print(f"\n----------------- Chargement des données {train_step} -------------")
    
    for class_name in classes:
        label_id = classes.index(class_name)
        path = os.path.join(data_dir, train_step, class_name)

        if not os.path.isdir(path):
            print(f"[ATTENTION] : le dossier {path} n'existe pas")
            return 
        
        file_list = [f for f in os.listdir(path) if f.lower().endswith('.jpg')]

        for filename in tqdm(file_list, desc=f"Traitement {class_name}"):
            img_path = os.path.join(path, filename)

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if img is not None:
                # hist_vector = calculate_histogram(img, m)
                hist_vector = calculate_LBP_histogram(img, m)
                X_features.append(hist_vector)
                y_labels.append(label_id)

    return np.array(X_features), np.array(y_labels)

def calculate_LBP_histogram(img, m=256):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 1
    n = 8 * radius
    lbp = local_binary_pattern(gray, n, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-7)
    return hist

def calculate_histogram(img, m):
    img = cv2.resize(img, (128, 128)) # Amélioration
    
    hist = np.zeros(3 * m)
    
    if img is None :
        return np.zeros(m * 3, dtype=np.float32)
    
    height, width, _ = img.shape
    
    for row in range(height):
        for col in range(width):
            pixel = img[row, col]
            hist[pixel[0]] += 1
            hist[int(pixel[1]) + m] += 1
            hist[int(pixel[2]) + (2 * m)] += 1
            
    size = height * width
    
    if size > 0:
        hist /= size

    return hist

def predict_knn(k, X_train, y_train, hist_test):
    
    # distance_array = np.sqrt(np.sum((X_train - hist_test)**2, axis=1))
    
    list_distance = []
    num_train_examples = X_train.shape[0]
    
    for row in range(num_train_examples):
        distance = 0.0
        hist_train = X_train[row]
        
        for col in range(hist_train.shape[0]):
            distance += (hist_train[col] - hist_test[col]) ** 2;
        distance = sqrt(distance)
        list_distance.append((distance, y_train[row]))
        
    list_distance.sort(key = lambda x : x[0])
    
    k_nearest_neighbors = list_distance[:k]
    # k_nearest_neighbors = np.argsort(distance_array)[:k]
    
    classes_occurence = {}
    
    for _, label in k_nearest_neighbors:
        classes_occurence[label] = classes_occurence.get(label, 1) + 1
        
    predicted_class = max(classes_occurence.items(), key=lambda item : item[1])[0]
    
    # neighbor_labels = y_train[k_nearest_neighbors]
    # predicted_class = np.argmax(np.bincount(neighbor_labels))
    
    return predicted_class

def evaluate_knn(dir_path, categories, K = 3):
    
    data = load_knn_data()
    if data is None:
        X_train, y_train = load_data(dir_path, categories, train_step="Training")
        X_test, y_test = load_data(dir_path, categories, train_step="Testing")
        save_knn_data(X_train, y_train, X_test, y_test)
    else:
        X_train, y_train, X_test, y_test = data
    
    num_classes = len(categories)
    num_test_examples = X_test.shape[0]
    
    C = np.zeros((num_classes, num_classes))
    
    print(f"\n--- Évaluation des {num_test_examples} échantillons de test (K={K}) ---")
    
    for row in tqdm(range(num_test_examples), desc="Prédiction des tests"):
        c_ground_truth = y_test[row]
        hist_test = X_test[row]
        c_predicted = predict_knn(K, X_train, y_train, hist_test)
        C[c_predicted, c_ground_truth] += 1
        
    correct_predictions = np.trace(C)
    total_predictions = np.sum(C)
    
    accuracy = correct_predictions / total_predictions
    
    return accuracy, C

def advanced_metrics(C):
    TP = C[1,1]
    FN = C[0, 1]
    FP = C[1, 0]
    
    if (TP + FN) == 0: 
        recall = 0.0
    else: 
        recall = TP / (TP + FN)
        
    if (TP + FP) == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)
        
    if (precision + recall) == 0:
        f1_score = 0.0
    else: 
        f1_score = 2 * (precision * recall) / (precision + recall)
    return recall, precision, f1_score

if __name__ == "__main__":
    K = 4
    # class_names = ["notumor", "glioma"] # Dataset 1
    class_names = ["no_tumor", "glioma_tumor"] # Dataset 2

    # data_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset") # Dataset 1
    data_path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri") # Dataset 2
    print("Path to dataset files:", data_path)

    print("\n" + "-"*60)
    print(f" KNN Classifier (k = {K}) ".center(60))
    print("=" * 60)

    print(f"| Paramètre K : {K}".ljust(59) + "|")
    print(f"| {len(class_names)} Classes : {', '.join(class_names)}".ljust(59) + "|")
    print("=" * 60)

    accuracy, C = evaluate_knn(data_path, class_names, 4)
    recall, precision, f1_score = advanced_metrics(C)
    correct_samples = np.trace(C)
    total_samples = np.sum(C)

    print("="*60)
    print("  RÉSUMÉ DES RÉSULTATS  ".center(60))
    print("="*60)
    print(f"| Accuracy : {accuracy:.4f} ({accuracy*100:.2f} %)".ljust(59) + "|")
    print(f"| Recall : {recall:.4f} ({recall*100:.2f} %)".ljust(59) + "|")
    print(f"| Precision : {precision:.4f} ({precision*100:.2f} %)".ljust(59) + "|")
    print(f"| F1-Score : {f1_score:.4f} ({f1_score*100:.2f} %)".ljust(59) + "|")
    print(f"| Prédictions Correctes : {correct_samples}/{total_samples}".ljust(59) + "|")
    print("="*60)

    print(f"-------------------------- FIN ---------------------------")





