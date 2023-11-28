import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from dataloader import PatientDatasetBinary

import torch
from torch.utils.data import DataLoader
import monai
from monai.networks.nets import resnet10

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


import torch.optim as optim
import torch.nn as nn
import datetime

# Load data
data_path = '/home/slt2870/Glioblastoma_Infillstration_Classification/processed_patient_data.npy'
labels_path = '/home/slt2870/Glioblastoma_Infillstration_Classification/labels.npy'
data = np.load(data_path, allow_pickle=True)
labels = np.load(labels_path, allow_pickle=True)


# Parameters
model_name = "resnet10_binary"
current_date = datetime.datetime.now().strftime('%Y%m%d')
lr = 1e-4
n_input_channels = 4
num_classes = 2
num_epochs = 10


def plot_curves(y_values1, y_values2, title, y_label, fold, curve_type):
    epochs = range(1, len(y_values1) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, y_values1, 'b', label='Training')
    plt.plot(epochs, y_values2, 'r', label='Validation')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{curve_type}_curves_fold_{fold}_{model_name}_{current_date}.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, fold):

    # Get unique labels from y_true
    unique_labels = np.unique(y_true)
    matrix = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Fold {fold}')
    plt.savefig(os.path.join(log_dir, f"confusion_matrix_fold_{fold}_{model_name}_{current_date}.png"))
    plt.close()



# Logging setup
log_dir = "/home/slt2870/Glioblastoma_Infillstration_Classification/logs"
log_file = f"{model_name}_{current_date}_lr{lr}_epochs{num_epochs}.txt"
log_path = os.path.join(log_dir, log_file)

save_model_dir = "/home/slt2870/Glioblastoma_Infillstration_Classification/model_weights"

if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=log_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Exclude 'nan' samples
valid_indices = [i for i, label in enumerate(labels) if label != 'nan']
data = data[valid_indices]
labels = labels[valid_indices]

kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(data, labels)):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    misclassified_patient_ids = []

    print("Train indices for Fold ", fold, " : ", train_idx)
    print("Validation indices for Fold ", fold, " : ", val_idx)

    train_data, train_labels = data[train_idx], labels[train_idx]
    val_data, val_labels = data[val_idx], labels[val_idx]
    
    print("Train labels for Fold ", fold, " : ", train_labels)
    print("Validation labels for Fold ", fold, " : ", val_labels)

    unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
    unique_val_labels, val_counts = np.unique(val_labels, return_counts=True)

    # Log or print the results
    logger.info(f"Fold {fold + 1} Training Data:")
    for label, count in zip(unique_train_labels, train_counts):
        logger.info(f"Class {label}: {count} samples")

    logger.info(f"Fold {fold + 1} Validation Data:")
    for label, count in zip(unique_val_labels, val_counts):
        logger.info(f"Class {label}: {count} samples")

    train_dataset = PatientDatasetBinary(train_data, train_labels)
    val_dataset = PatientDatasetBinary(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = resnet10(spatial_dims=3, n_input_channels=n_input_channels, num_classes=num_classes, pretrained=False)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # Modified this line
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()  # Modified this line

        train_losses.append(epoch_train_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        # Validation
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_preds = []

        for inputs, targets, patient_idxs in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # Modified this line
            epoch_val_loss += loss.item()

            _, predicted = outputs.max(1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()  # Modified this line
            all_labels.extend(targets.cpu().numpy())  # Modified this line
            all_preds.extend(predicted.cpu().numpy())

            if epoch == num_epochs - 1:
                for i, pred in enumerate(predicted):
                    if pred != targets[i]:  # Modified this line
                        misclassified_patient_ids.append(patient_idxs[i].item())

        val_losses.append(epoch_val_loss / len(val_loader))
        val_accuracies.append(100 * correct_val / total_val)

        logger.info(f"Fold {fold + 1}, Epoch {epoch + 1}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")
        logger.info(f"Training Accuracy: {train_accuracies[-1]:.2f}, Validation Accuracy: {val_accuracies[-1]:.2f}")

        if epoch == num_epochs - 1:
            logger.info(f"Misclassified Patient IDs for Fold {fold + 1}: {misclassified_patient_ids}")

    plot_curves(train_losses, val_losses, f'Training and Validation Loss for Fold {fold + 1}', 'Loss', fold + 1, 'loss')
    plot_curves(train_accuracies, val_accuracies, f'Training and Validation Accuracy for Fold {fold + 1}', 'Accuracy (%)', fold + 1, 'accuracy')
    unique_labels = np.unique(all_labels)
    plot_confusion_matrix(all_labels, all_preds, unique_labels, fold + 1)

    model_save_path = os.path.join(save_model_dir, f"model_fold_{fold + 1}.pth")
    torch.save(model.state_dict(), model_save_path)



logger.info('Finished Training')