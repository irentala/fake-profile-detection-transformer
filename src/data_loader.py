# src/data_loader.py

import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical operations
import os  # Library for operating system interactions
import glob  # Library for file pattern matching
import io  # Library for handling I/O operations
from torch.utils.data import Dataset  # PyTorch class for creating datasets

# Custom dataset class for keystroke data
class KeystrokeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # Initialize data
        self.labels = labels  # Initialize labels

    def __len__(self):
        return len(self.data)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # Return the data and label at the given index

# Function to preprocess files and remove problematic lines
def preprocess_file(file_path, expected_fields=9):
    cleaned_lines = []  # List to hold cleaned lines
    with open(file_path, 'r') as file:  # Open the file in read mode
        for line in file:  # Iterate over each line in the file
            if len(line.strip().split('\t')) == expected_fields:  # Check if the line has the expected number of fields
                cleaned_lines.append(line)  # Add the valid line to cleaned_lines
    return cleaned_lines  # Return the list of cleaned lines

# Function to load and preprocess data from multiple files in a folder
def load_and_preprocess_data(folder_path):
    all_data = []  # List to hold all feature data
    all_labels = []  # List to hold all labels

    # Iterate over all files in the folder that match the pattern
    for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
        try:
            cleaned_lines = preprocess_file(file_path)  # Preprocess the file to remove problematic lines
            if not cleaned_lines:  # Check if there are no valid lines
                print(f"No valid lines found in the file {file_path}. Skipping this file.")
                continue  # Skip to the next file
            df = pd.read_csv(io.StringIO(''.join(cleaned_lines)), delimiter='\t')  # Read the cleaned lines into a DataFrame
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")  # Print an error message if there's an issue reading the file
            continue  # Skip to the next file

        print(f"Columns in the dataset {file_path}: {df.columns}")  # Print the column names for debugging

        # Check if required columns exist
        if 'RELEASE_TIME' not in df.columns or 'PRESS_TIME' not in df.columns:
            print(f"Required columns 'RELEASE_TIME' and 'PRESS_TIME' are not present in the dataset {file_path}. Skipping this file.")
            continue  # Skip to the next file

        # Calculate and normalize features
        df['hold_time'] = df['RELEASE_TIME'] - df['PRESS_TIME']  # Calculate hold time
        df['interval'] = df.groupby('PARTICIPANT_ID')['PRESS_TIME'].diff().fillna(0)  # Calculate interval between key presses
        df['hold_time'] = (df['hold_time'] - df['hold_time'].mean()) / df['hold_time'].std()  # Normalize hold time
        df['interval'] = (df['interval'] - df['interval'].mean()) / df['interval'].std()  # Normalize interval

        features = df[['hold_time', 'interval']].values  # Extract feature values
        labels = df['PARTICIPANT_ID'].values  # Extract labels

        features = features.astype(np.float32)  # Convert features to float32

        all_data.append(features)  # Add features to the list
        all_labels.append(labels)  # Add labels to the list

    if not all_data or not all_labels:
        raise ValueError("No valid data found in the specified folder.")  # Raise an error if no valid data is found

    all_data = np.concatenate(all_data, axis=0)  # Concatenate all feature data
    all_labels = np.concatenate(all_labels, axis=0)  # Concatenate all labels

    num_features = all_data.shape[1]  # Get the number of features
    unique_labels = len(set(all_labels))  # Get the number of unique labels
    sequence_length = len(all_labels) // unique_labels  # Calculate the sequence length

    max_length = sequence_length  # Set the maximum length
    padded_data = []  # List to hold padded data
    for label in set(all_labels):
        label_data = all_data[all_labels == label]  # Get data for the current label
        if len(label_data) > max_length:
            label_data = label_data[:max_length]  # Truncate data if it exceeds max_length
        elif len(label_data) < max_length:
            pad_size = max_length - len(label_data)  # Calculate the padding size
            label_data = np.pad(label_data, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)  # Pad the data
        padded_data.append(label_data)  # Add padded data to the list

    data = np.stack(padded_data)  # Stack the padded data
    labels = np.repeat(list(set(all_labels)), max_length)  # Repeat labels to match the data length

    return data, labels  # Return the processed data and labels
