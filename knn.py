import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
import os

class_names = ["beach", "city", "desert", "forest", "landscape", "snow"]

def load_data(data_dir, classes, train_step):
    m = 256
    # X : ce sont les images
    X = []
    label_id = {}
    # y : Ce sont les labels
    y = []
    for c in classes:
        label_id[c] = classes.index(c)

    for c in classes:
        path = os.path.join(data_dir, train_step, classes)
        y = [i for i in range(len(classes))]

        if not os.path.isdir(path):
            print(f"[ATTENTION] : le répertoire {path} n'existe pas")
            return 

        for i in label_id.keys():
            for filename in os.listdir(path):
                img_path = os.path.join(path, classes[i], filename)

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                X.append(img)
                y.append(label_id[i])

    return np.array(X), np.array(y)


def calculate_histogram(img, m):
    hist = np.array(3 * m)
    
    if img is None:
        return np.zeros(m * 3)
    
    height, width, color = img.shape
    
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

def evaluate_knn():
    accuracy = 0
    return accuracy

def predict_knn():
    first_class = 0
    return first_class

