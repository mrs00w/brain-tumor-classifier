import torch as nn
import torch.functional as F
import torch
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import kagglehub
from math import *
from tqdm import tqdm
from knn import load_data

def save_cnn_model(model, filename="cnn_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Poids du CNN sauvegardés dans {filename}")

def load_cnn_model(model, filename="cnn_model.pth"):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
        model.eval() # Important : met le modèle en mode évaluation
        print(f"Poids du CNN chargés depuis {filename}")
        return model
    return None

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1), # Les IRM sont souvent en gris
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def prepare_cnn_data(data_path):
    train_data = datasets.ImageFolder(root=os.path.join(data_path, 'Training'), transform=transform)
    test_data = datasets.ImageFolder(root=os.path.join(data_path, 'Testing'), transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()