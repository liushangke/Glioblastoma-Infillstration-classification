import os
import glob
import nrrd
import nibabel as nib
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.ndimage import binary_opening
from scipy.ndimage import zoom as resize

from dataloader import PatientDataset

import torch
from torch.utils.data import Dataset, DataLoader
import monai
from monai.networks.nets import resnet10

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import torch.optim as optim
import torch.nn as nn

import os
import datetime
import logging


data = np.load('/home/slt2870/Glioblastoma_Infillstration_Classification/processed_patient_data.npy', allow_pickle=True)
labels = np.load('/home/slt2870/Glioblastoma_Infillstration_Classification/labels.npy', allow_pickle=True)


# Parameters
model_name = "resnet10"
current_date = datetime.datetime.now().strftime('%Y%m%d')
lr = 1e-4
n_input_channels = 4
num_classes = 3
num_epochs = 10

# Define log directory and filename
log_dir = "/home/slt2870/Glioblastoma_Infillstration_Classification/logs"
log_file = f"{model_name}_{current_date}_lr{lr}_epochs{num_epochs}.txt"
log_path = os.path.join(log_dir, log_file)

save_model_dir = "/home/slt2870/Glioblastoma_Infillstration_Classification/model_weights"
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)


# Check and create log directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(filename=log_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # Also log to console

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up 4-fold stratified cross-validation
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(data, labels)):
    
    train_data, train_labels = data[train_idx], labels[train_idx]
    test_data, test_labels = data[test_idx], labels[test_idx]
    
    train_dataset = PatientDataset(train_data, train_labels, patches_per_sample=10)
    test_dataset = PatientDataset(test_data, test_labels, patches_per_sample=10)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # Initialize the ResNet model
    model = resnet10(spatial_dims=3, n_input_channels=n_input_channels, num_classes=num_classes, pretrained=False)
    model.to(device)  # Transfer model to GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets, patient_idxs in train_loader:  # Updated this line
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(targets, dim=1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Training Loss: {epoch_loss/len(train_loader):.4f}")
        
    # Test loop
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    misclassified_patient_ids = []
    
    with torch.no_grad():
        for inputs, targets, patient_idxs in test_loader:  # Assume dataset returns patient_idxs
            inputs, targets = inputs.to(device), targets.to(device)  # Transfer data to GPU
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_samples += targets.size(0)
            total_correct += (predicted == torch.argmax(targets, dim=1)).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(torch.argmax(targets, dim=1).cpu().numpy())
            
            # Track misclassified samples
            for i, pred in enumerate(predicted):
                if pred != torch.argmax(targets[i], dim=0):
                    misclassified_patient_ids.append(patient_idxs[i].item())
    
    accuracy = total_correct / total_samples
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Fold {fold + 1}, Test Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
    print(f"Misclassified Patient IDs in Fold {fold + 1}: {misclassified_patient_ids}")

    # Save model weights
    model_save_path = os.path.join(save_model_dir, f"model_fold_{fold + 1}.pth")
    torch.save(model.state_dict(), model_save_path)

print('Finished Training')
