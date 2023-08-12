import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import PatientDataset
from model.unet import UNet3DPatchBased
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter

# Define parameters
lr = 1e-4
batch_size = 2
num_epochs = 50
in_channel = 4
out_channel = 3

# Load your data and labels
all_patient_data = np.load('processed_patient_data.npy', allow_pickle=True)
label_dict = np.load('labels.npy', allow_pickle=True)

# Create your dataset and data loader
dataset = PatientDataset(all_patient_data, label_dict)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Determine the device to run the model on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model, loss function, and optimizer
model = UNet3DPatchBased(in_channels=in_channel, out_channels=out_channel).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Create a SummaryWriter object
writer = SummaryWriter()

# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    epoch_misclassified = []

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and correct predictions
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels_max = torch.max(labels.data, 1)
        correct = (predicted == labels_max).sum().item()
        epoch_correct += correct
        epoch_total += labels.size(0)

        # Track misclassified samples
        misclassified_indices = (predicted != labels_max).nonzero(as_tuple=True)[0]
        epoch_misclassified.extend(misclassified_indices.tolist())

        # Track precision, recall, and F1-score
        precision = precision_score(labels_max.cpu(), predicted.cpu(), average='macro')
        recall = recall_score(labels_max.cpu(), predicted.cpu(), average='macro')
        f1 = f1_score(labels_max.cpu(), predicted.cpu(), average='macro')

    epoch_loss /= len(dataloader)
    epoch_acc = epoch_correct / epoch_total
    writer.add_scalar('Training/Loss', epoch_loss, epoch)
    writer.add_scalar('Training/Accuracy', epoch_acc, epoch)
    writer.add_scalar('Training/Precision', precision, epoch)
    writer.add_scalar('Training/Recall', recall, epoch)
    writer.add_scalar('Training/F1', f1, epoch)

    # Add histograms for model's weights and gradients
    for name, param in model.named_parameters():
        writer.add_histogram('Weights/' + name, param.data, epoch)
        if param.grad is not None:
            writer.add_histogram('Gradients/' + name, param.grad, epoch)

    # Add images for some misclassified samples if your data are images
    if epoch_misclassified:
        # Save the misclassified 3D images for visualization
        wrong_images = [dataset[i][0] for i in epoch_misclassified[:10]]
        # Save them to disk, or visualize directly here
        for idx, img in enumerate(wrong_images):
            # convert tensor to numpy array
            img_np = img.cpu().numpy()
            # take the slice in the middle of the 3D image
            slice_idx = img_np.shape[-1] // 2
            img_slice = img_np[..., slice_idx]
            # normalize to [0, 1] for visualization
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            # write the image slice to tensorboard
            writer.add_image(f"Misclassified/Image_{idx}", img_slice, epoch, dataformats='HW')

writer.close()

print('Finished Training')
