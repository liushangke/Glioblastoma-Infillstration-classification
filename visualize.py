import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataloader import PatientDataset  # Replace with your dataloader module

def normalize_image(image):
    min_val = torch.min(image)
    max_val = torch.max(image)
    return (image - min_val) / (max_val - min_val)

def main():
    all_patient_data = np.load('processed_patient_data.npy', allow_pickle=True)
    label_dict = np.load('label_dict.npy', allow_pickle=True).item()

    dataset = PatientDataset(all_patient_data, label_dict)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Load the first batch and visualize the first example
    inputs, labels = next(iter(dataloader))

    # Select a range of slices in the middle
    slice_idxs = range(inputs.size(3) // 2 - 5, inputs.size(3) // 2 + 5)

    sequence_names = ['T1', 'T1c', 'T2', 'FLAIR']
    for i in range(4):
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f"Slices of {sequence_names[i]} sequence")
        for slice_idx, ax in zip(slice_idxs, axs.flatten()):
            image = normalize_image(inputs[0, i, :, :, slice_idx])  # Normalize image before plotting
            ax.imshow(image, cmap='gray')
            ax.axis('off')
            ax.set_title(f"Slice {slice_idx}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

