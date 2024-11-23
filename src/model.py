import torch
import torch.nn as nn

# Transformer-based encoder model
class TransformerKeystrokeEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(TransformerKeystrokeEncoder, self).__init__()
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

# Bi-Encoder model
class BiEncoderModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(BiEncoderModel, self).__init__()
        self.encoder_a = TransformerKeystrokeEncoder(input_dim, model_dim, num_heads, num_layers)  # Encoder for Keystroke A
        self.encoder_b = TransformerKeystrokeEncoder(input_dim, model_dim, num_heads, num_layers)  # Encoder for Keystroke B

    def forward(self, keystroke_a, keystroke_b):
        embedding_a = self.encoder_a(keystroke_a)  # Get embedding for Keystroke A
        embedding_b = self.encoder_b(keystroke_b)  # Get embedding for Keystroke B
        return embedding_a, embedding_b
