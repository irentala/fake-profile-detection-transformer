# src/train.py

import torch  # PyTorch library for tensor operations and neural networks
from src.model import TransformerKeystrokeModel, TripletLoss  # Import model and loss function

# Training loop
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize total loss
    for data, labels in dataloader:  # Iterate over batches of data
        data, labels = data.to(device), labels.to(device)  # Move data and labels to the specified device
        data = data.float()  # Convert data to float32
        optimizer.zero_grad()  # Clear the gradients
        anchor, positive, negative = data[:, 0, :], data[:, 1, :], data[:, 2, :]  # Split data into anchor, positive, and negative
        anchor_out = model(anchor)  # Compute the model output for anchor samples
        positive_out = model(positive)  # Compute the model output for positive samples
        negative_out = model(negative)  # Compute the model output for negative samples
        loss = criterion(anchor_out, positive_out, negative_out)  # Compute the triplet loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters
        total_loss += loss.item()  # Accumulate the total loss
    return total_loss / len(dataloader)  # Return the average loss for the epoch
