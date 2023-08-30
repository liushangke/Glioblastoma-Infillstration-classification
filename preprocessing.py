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

def data_investigation(all_patient_data, all_labels):
    # Create a dictionary to store the counts of each unique shape
    shape_counts = defaultdict(int)
    patient_shapes = {}  # Dictionary to store shapes for each patient
    
    # Iterate over all the patient data
    for idx, patient in enumerate(all_patient_data):
        # You can choose any sequence, since they all have the same shape
        sequence_data = patient['T1-3D.nii']
        
        # Count the shape of the sequence data
        shape_counts[sequence_data.shape] += 1
        patient_shapes[patient['patient_id']] = sequence_data.shape

    return shape_counts, patient_shapes


def plot_shape_counts(shape_counts, top_n=10):
    # Sort the shapes by frequency in descending order and keep only the top N
    sorted_shapes = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Convert the keys (shapes) to strings so they can be used as labels
    labels = range(1, len(sorted_shapes)+1)
    counts = [count//4 for shape, count in sorted_shapes]

    # Create a new figure with a larger size
    plt.figure(figsize=(20, 10))

    # Create the bar plot
    bars = plt.bar(labels, counts)

    # Add text above each bar
    for i, (shape, count) in enumerate(sorted_shapes):
        plt.text(i+1, count//4, str(shape), ha='center', va='bottom')

    plt.title('Distribution of Top {} Sequence Shapes'.format(top_n), fontsize=16)
    plt.xlabel('Shape Index', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(labels, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Display the plot
    plt.show()


def prepare_data_for_plotting(shape_counts):
    widths, heights, depths, frequencies = [], [], [], []
    for shape, count in shape_counts.items():
        width, height, depth = shape
        widths.append(width)
        heights.append(height)
        depths.append(depth)
        frequencies.append(count)
    return widths, heights, depths, frequencies


def round_to_power_of_two(n):
    return 2**np.round(np.log2(n))

def plot_3d_shape_distribution(shape_counts):
    # Prepare the data for plotting
    rounded_shapes = {(round_to_power_of_two(w), round_to_power_of_two(h), round_to_power_of_two(d)): c 
                      for (w, h, d), c in shape_counts.items()}
    sorted_shapes = sorted(rounded_shapes.items(), key=lambda x: x[1], reverse=True)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define a color map
    cmap = plt.get_cmap("viridis")

    for (w, h, d), c in sorted_shapes:
        # Define the vertices of the cube
        Z = np.array([[0, 0, 0],
                      [w, 0, 0],
                      [w, h, 0],
                      [0, h, 0],
                      [0, 0, d],
                      [w, 0, d],
                      [w, h, d],
                      [0, h, d]])

        # Define the edges of the cube
        edges = [[Z[j], Z[k]] for j, k in zip([0, 1, 5, 6, 4, 2, 3, 7], [1, 5, 6, 2, 0, 3, 7, 4])]

        # Create the 3D surface
        faces = Poly3DCollection(edges, linewidths=1, edgecolors='r', alpha=0.2)
        faces.set_facecolor(cmap(c/np.max(list(shape_counts.values()))))

        ax.add_collection3d(faces)

    # Set the labels and title
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Depth')
    ax.set_title('3D Shape Distribution')

    plt.show()

    # Print the most common rounded shape
    most_common_shape = sorted_shapes[0][0]
    print(f"The most common rounded shape is {most_common_shape}")




def load_data(mri_path, seg_path):
    mri_img = nib.load(mri_path)
    mri_data = mri_img.get_fdata()
    seg_data, _ = nrrd.read(seg_path)
    return mri_data, seg_data


def extract_roi(mri_data, seg_data):
    return mri_data * seg_data


def process_patient_folder(patient_folder):
    sequence_files = {}
    sequence_types = ['T1-3D.nii', 'T1c-3D.nii', 'T2-3D.nii', 'FLAIR-3D.nii']

    # Get paths to the MRI data files
    for sequence_type in sequence_types:
        sequence_files[sequence_type] = glob.glob(
            os.path.join(patient_folder, sequence_type))
        if not sequence_files[sequence_type]:
            print(f"Missing {sequence_type} in folder: {patient_folder}")
            return None

    # Get path to the segmentation file
    seg_files = glob.glob(os.path.join(patient_folder, '*.nrrd'))
    seg_pattern = re.compile(r'segmentation\.seg', re.IGNORECASE)
    seg_file = None

    for file in seg_files:
        if seg_pattern.search(file):
            seg_file = file
            break

    if not seg_file:
        print(
            f"Missing segmentation.seg.nrrd file in folder: {patient_folder}")
        return None

    rois = {}
    # Load MRI data and segmentation for each sequence and extract ROI
    for sequence_type in sequence_types:
        mri_data, seg_data = load_data(
            sequence_files[sequence_type][0], seg_file)
        roi = extract_roi(mri_data, seg_data)
        # print(f'ROI size for sequence {sequence_type}: {roi.shape}')
        rois[sequence_type] = roi

    return rois


def process_all_patients(data_dir, label_dict):
    # Get a list of all patient folders in data_dir
    patient_folders = [f for f in os.listdir(
        data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Sort the patient_folders list to ensure it's in the same order as the labels
    patient_folders.sort(key=lambda x: int(x))

    all_patient_data = []
    all_labels = []

    for folder in patient_folders:
        patient_id = int(folder)  # Convert folder name to integer
        patient_folder = os.path.join(data_dir, folder)

        patient_data = process_patient_folder(patient_folder)
        if patient_data is not None:
            # Add the patient_id to the patient_data dictionary
            patient_data['patient_id'] = patient_id
            all_patient_data.append(patient_data)
            all_labels.append(label_dict[patient_id])

    print(
        f"Total patient folders: {len(patient_folders)}, successfully processed: {len(all_patient_data)}")
    return all_patient_data, all_labels


# Function to visualize 3D ROI
def visualize_3d_roi(roi, sequence_type, threshold=0.1):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define coordinates
    x, y, z = np.where(roi > threshold)

    ax.scatter(x, y, z, c='red', alpha=0.6, s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def create_label_dict(label_excel_path):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(label_excel_path)

    # Create a dictionary mapping patient names to labels
    label_dict = df.set_index('Subject number')['Infiltration'].to_dict()

    return label_dict


def main():
    # Specify the directories where your raw data and labels are stored
    data_dir = "/home/slt2870/Glioblastoma_Infillstration_Classification/data/Brainstem Annotations"  # Replace with your path
    # Replace with your path
    label_dir = "/home/slt2870/Glioblastoma_Infillstration_Classification/data/Deidentified Brainstem_dicoms full set with hashed IDs.xlsx"

    label_dict = create_label_dict(label_dir)

    all_patient_data, all_labels = process_all_patients(data_dir, label_dict)

    shape_counts, patient_shapes = data_investigation(all_patient_data, all_labels)
    plot_shape_counts(shape_counts)
    
    # Print out the shapes for each patient
    print("\nShapes for each patient:")
    for patient_id, shape in patient_shapes.items():
        print(f"Patient {patient_id}: Shape {shape}")

    np.save('processed_patient_data.npy', all_patient_data)
    np.save('labels.npy', all_labels)


if __name__ == "__main__":
    main()
