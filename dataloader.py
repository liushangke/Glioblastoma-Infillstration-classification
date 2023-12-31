import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom as resize
import nibabel as nib

class PatientDataset(Dataset):
    def __init__(self, all_patient_data, labels_array, patches_per_sample=10, resize_shape=(100, 100, 100)):
        self.data = all_patient_data
        # Replicate each label 'patches_per_sample' times to match the number of patches
        self.labels = [self.map_label(labels_array[patient_idx]) for patient_idx, _ in enumerate(all_patient_data) for _ in range(patches_per_sample)]
        self.patch_size = (16, 16, 16)  # Desired patch size
        self.patches_per_sample = patches_per_sample  # Number of patches to extract per sample
        self.resize_shape = resize_shape
        self.patient_ids = [patient['patient_id'] for patient in all_patient_data]


    def map_label(self, label):
        if label == "Extensive":
            return [1, 0, 0]
        elif label == "None":
            return [0, 1, 0]
        else:
            return [0, 0, 1]

    # def map_label(self, label):
    #     return 1 if label == "Extensive" else 0

        
    def inverse_map_label(self, one_hot_label):
        label_map = {
            "Extensive": [1, 0, 0],
            "None": [0, 1, 0],
            "Microscopic": [0, 0, 1]
        }
        for label, one_hot in label_map.items():
            if one_hot == one_hot_label:
                return label
   

    def resize_sequence(self, sequence):
        # Use scipy's zoom to resize the sequence based on the resize_shape attribute
        factors = (self.resize_shape[0]/sequence.shape[0], 
                   self.resize_shape[1]/sequence.shape[1], 
                   self.resize_shape[2]/sequence.shape[2])
        return resize(sequence, factors)

                                 
    def __len__(self):
        return len(self.data) * self.patches_per_sample

    def __getitem__(self, index):
        # Determine the actual patient sample and patch index
        patient_index = index // self.patches_per_sample
        patient_id = self.patient_ids[patient_index]
        patch_idx = index % self.patches_per_sample  # This is not used but is here for clarity
        
        patient_data = self.data[patient_index]

        # Process the sequences
        sequence_types = ['T1-3D.nii', 'T1c-3D.nii', 'T2-3D.nii', 'FLAIR-3D.nii']
        sequences = []
        for sequence_type in sequence_types:
            tensor = patient_data[sequence_type].astype(np.float32)
            tensor = self.resize_sequence(tensor)
            
            # Randomly select a starting point for the patch
            start_x = np.random.randint(0, tensor.shape[0] - self.patch_size[0])
            start_y = np.random.randint(0, tensor.shape[1] - self.patch_size[1])
            start_z = np.random.randint(0, tensor.shape[2] - self.patch_size[2])

            # Extract the patch
            patch = tensor[start_x:start_x + self.patch_size[0],
                           start_y:start_y + self.patch_size[1],
                           start_z:start_z + self.patch_size[2]]

            sequences.append(torch.from_numpy(patch))

        # Stack the sequences into a single tensor
        tensor = torch.stack(sequences).squeeze(1)
        label = torch.tensor(self.labels[patient_index], dtype=torch.float32)

        return tensor, label, patient_id

# class PatientDatasetBinary(Dataset):
#     def __init__(self, all_patient_data, labels_array, patches_per_sample=10, resize_shape=(100, 100, 100)):
#         self.data = all_patient_data
#         # Replicate each label 'patches_per_sample' times to match the number of patches
#         self.labels = [self.map_label(labels_array[patient_idx]) for patient_idx, _ in enumerate(all_patient_data) for _ in range(patches_per_sample)]
#         self.patch_size = (16, 16, 16)  # Desired patch size
#         self.patches_per_sample = patches_per_sample  # Number of patches to extract per sample
#         self.resize_shape = resize_shape
#         self.patient_ids = [patient['patient_id'] for patient in all_patient_data]

#     def map_label(self, label):
#         return 1 if label == "Extensive" else 0

#     def resize_sequence(self, sequence):
#         # Use scipy's zoom to resize the sequence based on the resize_shape attribute
#         factors = (self.resize_shape[0]/sequence.shape[0], 
#                    self.resize_shape[1]/sequence.shape[1], 
#                    self.resize_shape[2]/sequence.shape[2])
#         return resize(sequence, factors)
                                 
#     def __len__(self):
#         return len(self.data) * self.patches_per_sample

#     def __getitem__(self, index):
#         # Determine the actual patient sample and patch index
#         patient_index = index // self.patches_per_sample
#         patient_id = self.patient_ids[patient_index]
#         patch_idx = index % self.patches_per_sample  # This is not used but is here for clarity
        
#         patient_data = self.data[patient_index]

#         # Process the sequences
#         sequence_types = ['T1-3D.nii', 'T1c-3D.nii', 'T2-3D.nii', 'FLAIR-3D.nii']
#         sequences = []
#         for sequence_type in sequence_types:
#             tensor = patient_data[sequence_type].astype(np.float32)
#             tensor = self.resize_sequence(tensor)
            
#             # Randomly select a starting point for the patch
#             start_x = np.random.randint(0, tensor.shape[0] - self.patch_size[0])
#             start_y = np.random.randint(0, tensor.shape[1] - self.patch_size[1])
#             start_z = np.random.randint(0, tensor.shape[2] - self.patch_size[2])

#             # Extract the patch
#             patch = tensor[start_x:start_x + self.patch_size[0],
#                            start_y:start_y + self.patch_size[1],
#                            start_z:start_z + self.patch_size[2]]

#             sequences.append(torch.from_numpy(patch))

#         # Stack the sequences into a single tensor
#         tensor = torch.stack(sequences).squeeze(1)
#         label = torch.tensor(self.labels[patient_index], dtype=torch.long)  # Ensure the label is of type long

#         return tensor, label, patient_id

class PatientDatasetBinary(Dataset):
    def __init__(self, all_patient_data, labels_array, resize_shape=(100, 100, 100), augment=True):
        self.data = all_patient_data
        self.labels = [self.map_label(label) for label in labels_array]
        self.resize_shape = resize_shape
        self.patient_ids = [patient['patient_id'] for patient in all_patient_data]
        self.augment = augment

        # Define augmentation transforms using torchio
        self.transforms = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2)),
            tio.RandomAffine(degrees=(10, 10, 10), scales=(0.9, 1.1)),
            tio.RandomNoise(std=0.5),
            tio.RandomElasticDeformation(),
        ])

    def map_label(self, label):
        return 1 if label == "Extensive" else 0

    def resize_sequence(self, sequence):
        # Use scipy's zoom to resize the sequence based on the resize_shape attribute
        factors = (self.resize_shape[0] / sequence.shape[0], 
                   self.resize_shape[1] / sequence.shape[1], 
                   self.resize_shape[2] / sequence.shape[2])
        return resize(sequence, factors)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        patient_data = self.data[index]
        patient_id = self.patient_ids[index]

        # Process the sequences
        sequence_types = ['T1-3D.nii', 'T1c-3D.nii', 'T2-3D.nii', 'FLAIR-3D.nii']
        sequences = []
        for sequence_type in sequence_types:
            tensor = patient_data[sequence_type].astype(np.float32)
            tensor = self.resize_sequence(tensor)
            
            # Apply augmentation
            if self.augment:
                subject = tio.Subject(mri=tio.ScalarImage(tensor=tensor))
                transformed_subject = self.transforms(subject)
                tensor = transformed_subject['mri']['data'].numpy()

            sequences.append(torch.from_numpy(tensor))

        # Stack the sequences into a single tensor
        tensor = torch.stack(sequences)
        label = torch.tensor(self.labels[index], dtype=torch.long)  # Ensure the label is of type long

        return tensor, label, patient_id


def display_dataset_info(dataset):
    # Total Length of the Dataset
    print(f"Total number of patches in the dataset: {len(dataset)}")
    
    # Label Distribution
    label_counts = {}
    for label in dataset.labels:
        label_str = str(label)  # Convert list to string for easy counting
        label_counts[label_str] = label_counts.get(label_str, 0) + 1
    
    print("\nLabel Distribution:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    
    # Sample Information
    sample_data, sample_label = dataset[0]  # Extract the first sample as an example
    print("\nSample Information:")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample label: {sample_label}")



def main():
    all_patient_data = np.load('Glioblastoma_Infillstration_Classification/processed_patient_data.npy', allow_pickle=True)
    patches_per_sample = 10
    labels_array = np.load('Glioblastoma_Infillstration_Classification/labels.npy', allow_pickle=True)
    dataset = PatientDataset(all_patient_data, labels_array, patches_per_sample=patches_per_sample)

    display_dataset_info(dataset)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Print the sizes of the volumes to verify that patch extraction is working correctly
    for i, (inputs, labels) in enumerate(data_loader):
        print(f"Batch {i + 1}: Size of inputs: {inputs.size()}")
        if i == 3:  # This will print the sizes for the first 4 batches
            break

if __name__ == "__main__":
    main()
