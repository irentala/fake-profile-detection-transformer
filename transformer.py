import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import numpy as np
import glob
import io  # Importing StringIO from the standard library


# Custom dataset class for keystroke data
class KeystrokeDataset(Dataset):
    def __init__(self, data, labels):
        # Initialize the dataset with data and labels
        self.data = data
        self.labels = labels

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return the data and label at the given index
        return self.data[idx], self.labels[idx]


# Function to preprocess files and remove problematic lines
def preprocess_file(file_path, expected_fields=9):
    cleaned_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if len(line.strip().split('\t')) == expected_fields:
                cleaned_lines.append(line)
    return cleaned_lines


# Function to load and preprocess data from multiple files in a folder
def load_and_preprocess_data(folder_path):
    all_data = []
    all_labels = []

    # Iterate over all files in the folder
    for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
        try:
            # Preprocess the file to remove problematic lines
            cleaned_lines = preprocess_file(file_path)
            if not cleaned_lines:
                print(f"No valid lines found in the file {file_path}. Skipping this file.")
                continue
            # Read the cleaned lines into a Pandas DataFrame
            df = pd.read_csv(io.StringIO(''.join(cleaned_lines)), delimiter='\t')
        except Exception as e:
            # Print an error message if there is an issue reading the file and skip this file
            print(f"Error reading file {file_path}: {e}")
            continue

        # Print the column names to debug
        print(f"Columns in the dataset {file_path}: {df.columns}")

        # Check if 'RELEASE_TIME' and 'PRESS_TIME' columns exist
        if 'RELEASE_TIME' not in df.columns or 'PRESS_TIME' not in df.columns:
            print(
                f"Required columns 'RELEASE_TIME' and 'PRESS_TIME' are not present in the dataset {file_path}. Skipping this file.")
            continue

        # Extract features (example)
        df['hold_time'] = df['RELEASE_TIME'] - df['PRESS_TIME']  # Calculate hold time
        df['interval'] = df.groupby('PARTICIPANT_ID')['PRESS_TIME'].diff().fillna(
            0)  # Calculate interval between key presses

        # Normalize features
        df['hold_time'] = (df['hold_time'] - df['hold_time'].mean()) / df['hold_time'].std()  # Normalize hold time
        df['interval'] = (df['interval'] - df['interval'].mean()) / df['interval'].std()  # Normalize interval

        # Convert features and labels to numpy arrays
        features = df[['hold_time', 'interval']].values
        labels = df['PARTICIPANT_ID'].values

        # Convert features to float32
        features = features.astype(np.float32)

        # Add the data and labels to the lists
        all_data.append(features)
        all_labels.append(labels)

    if not all_data or not all_labels:
        raise ValueError("No valid data found in the specified folder.")

    # Concatenate all the data and labels
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Reshape data to (num_samples, sequence_length, num_features)
    num_features = all_data.shape[1]
    unique_labels = len(set(all_labels))
    sequence_length = len(all_labels) // unique_labels

    # Padding or truncating the data to ensure consistent sequence length
    max_length = sequence_length
    padded_data = []
    for label in set(all_labels):
        label_data = all_data[all_labels == label]
        if len(label_data) > max_length:
            label_data = label_data[:max_length]
        elif len(label_data) < max_length:
            pad_size = max_length - len(label_data)
            label_data = np.pad(label_data, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        padded_data.append(label_data)

    data = np.stack(padded_data)
    labels = np.repeat(list(set(all_labels)), max_length)

    return data, labels


# Transformer-based model
class TransformerKeystrokeModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(TransformerKeystrokeModel, self).__init__()
        # Embedding layer to project input features to a higher dimension
        self.embedding = nn.Linear(input_dim, model_dim)
        # Transformer encoder layer with specified dimensions and heads, using batch_first=True
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                          batch_first=True)
        # Fully connected layer to output the final representation
        self.fc = nn.Linear(model_dim, model_dim)
        # Adaptive average pooling to reduce the sequence dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Apply the embedding layer
        x = self.embedding(x)
        # Apply the transformer encoder
        x = self.transformer(x, x)
        # Apply the fully connected layer
        x = self.fc(x)
        # Ensure the tensor has 3 dimensions before transposing
        if x.dim() == 3:
            # Transpose the tensor for pooling
            x = x.transpose(1, 2)
        # Apply pooling and squeeze the last dimension
        x = self.pool(x).squeeze(-1)
        return x


# Triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute the pairwise distance between anchor and positive samples
        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive)
        # Compute the pairwise distance between anchor and negative samples
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative)
        # Compute the triplet loss
        loss = torch.mean(torch.relu(pos_dist - neg_dist + self.margin))
        return loss


# Training loop
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize total loss
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)  # Move data and labels to the specified device
        data = data.float()  # Convert data to float32
        optimizer.zero_grad()  # Clear the gradients
        anchor, positive, negative = data[:, 0, :], data[:, 1, :], data[:, 2,
                                                                   :]  # Split data into anchor, positive, and negative
        anchor_out = model(anchor)  # Compute the model output for anchor samples
        positive_out = model(positive)  # Compute the model output for positive samples
        negative_out = model(negative)  # Compute the model output for negative samples
        loss = criterion(anchor_out, positive_out, negative_out)  # Compute the triplet loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters
        total_loss += loss.item()  # Accumulate the total loss
    return total_loss / len(dataloader)  # Return the average loss for the epoch


# Evaluation function
def evaluate_model(model, known_genuine_embeddings, test_dataloader, threshold, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_dataloader:
            data, labels = data.to(device), labels.to(device)  # Move data and labels to the specified device
            data = data.float()  # Convert data to float32
            embeddings = model(data)  # Compute embeddings for test sequences
            for embedding in embeddings:
                # Compute distances to known genuine embeddings
                distances = [
                    torch.nn.functional.pairwise_distance(embedding.unsqueeze(0), genuine_embedding.unsqueeze(0)).item()
                    for genuine_embedding in known_genuine_embeddings]
                min_distance = min(distances)
                # Classify as genuine if distance is below the threshold, otherwise as imposter
                if min_distance < threshold:
                    correct += 1
                total += 1
    accuracy = correct / total
    print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy:.4f}")
    return accuracy


# Main execution
if __name__ == '__main__':
    # Set the folder path
    folder_path = '/Users/irentala/PycharmProjects/GPTTransformerkeystroke/Aaltodesktopkeystrokedataset/Samplesize'

    # Load dataset
    data, labels = load_and_preprocess_data(folder_path)

    # Prepare dataset and dataloader
    dataset = KeystrokeDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, criterion, optimizer
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available, otherwise CPU
    model = TransformerKeystrokeModel(input_dim=data.shape[-1], model_dim=128, num_heads=8, num_layers=6).to(
        device)  # Initialize the model
    criterion = TripletLoss(margin=1.0)  # Initialize the triplet loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Initialize the optimizer

    # Training the model
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_model(model, dataloader, optimizer, criterion, device)  # Train the model for one epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")  # Print the loss for the current epoch

    print("Training complete.")

    # Evaluate the model
    # Generate embeddings for known genuine sequences (for simplicity, using the same dataset here)
    known_genuine_embeddings = []
    model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device).float()
            embeddings = model(data)
            known_genuine_embeddings.extend(embeddings)

    # Prepare test dataloader (for simplicity, using the same dataset here)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    threshold = 0.5  # Example threshold, adjust based on your data and requirements
    accuracy = evaluate_model(model, known_genuine_embeddings, test_dataloader, threshold, device)
    print(f"Evaluation accuracy: {accuracy:.4f}")
