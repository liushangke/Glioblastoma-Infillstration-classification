import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom as resize
import nibabel as nib

class PatientDataset(Dataset):
    def __init__(self, all_patient_data, label_dict):
        self.data = all_patient_data
        self.labels = [self.map_label(label_dict[patient['patient_id']]) for patient in all_patient_data]
        self.rescale_sizes = (64, 64, 64)  # Desired size after resizing

    def map_label(self, label):
        if label == "Extensive":
            return [1, 0, 0]
        elif label == "None":
            return [0, 1, 0]
        else:
            return [0, 0, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        patient_data = self.data[index]

        # Process the sequences
        sequence_types = ['T1-3D.nii', 'T1c-3D.nii', 'T2-3D.nii', 'FLAIR-3D.nii']
        sequences = []
        for sequence_type in sequence_types:
            tensor = patient_data[sequence_type].astype(np.float32)
            resize_factors = tuple(new_dim / old_dim for new_dim, old_dim in zip(self.rescale_sizes, tensor.shape))
            tensor = resize(tensor, resize_factors, mode='constant')
            sequences.append(torch.from_numpy(tensor))

        # Stack the sequences into a single tensor
        tensor = torch.stack(sequences).squeeze(1)

        label = torch.tensor(self.labels[index], dtype=torch.float32)

        return tensor, label

def main():
    all_patient_data = np.load('processed_patient_data.npy', allow_pickle=True)
    label_dict = np.load('label_dict.npy', allow_pickle=True).item()

    dataset = PatientDataset(all_patient_data, label_dict)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Print the sizes of the volumes to verify that resizing worked correctly
    for i, (inputs, labels) in enumerate(data_loader):
        print(f"Batch {i + 1}: Size of inputs: {inputs.size()}")
        if i == 3:
            break

if __name__ == "__main__":
    main()
