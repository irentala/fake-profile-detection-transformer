# src/model.py

import torch  # PyTorch library for tensor operations and neural networks
import torch.nn as nn  # PyTorch module for neural network layers

# Transformer-based model
class TransformerKeystrokeModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(TransformerKeystrokeModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)  # Embedding layer to project input features to a higher dimension
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, batch_first=True)  # Transformer encoder
        self.fc = nn.Linear(model_dim, model_dim)  # Fully connected layer
        self.pool = nn.AdaptiveAvgPool1d(1)  # Adaptive average pooling layer

    def forward(self, x):
        x = self.embedding(x)  # Apply the embedding layer
        x = self.transformer(x, x)  # Apply the transformer encoder
        x = self.fc(x)  # Apply the fully connected layer
        if x.dim() == 3:
            x = x.transpose(1, 2)  # Transpose the tensor for pooling
        x = self.pool(x).squeeze(-1)  # Apply pooling and squeeze the last dimension
        return x  # Return the output

# Triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin  # Margin for the triplet loss

    def forward(self, anchor, positive, negative):
        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive)  # Pairwise distance between anchor and positive
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative)  # Pairwise distance between anchor and negative
        loss = torch.mean(torch.relu(pos_dist - neg_dist + self.margin))  # Compute the triplet loss
        return loss  # Return the loss
