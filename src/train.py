import torch
import torch.nn as nn


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

        # Compute the model output for anchor, positive, and negative samples separately
        anchor_out, _ = model(anchor, anchor)  # Get embedding for anchor
        positive_out, _ = model(positive, positive)  # Get embedding for positive
        negative_out, _ = model(negative, negative)  # Get embedding for negative

        loss = criterion(anchor_out, positive_out, negative_out)  # Compute the triplet loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters
        total_loss += loss.item()  # Accumulate the total loss
    return total_loss / len(dataloader)  # Return the average loss for the epoch
