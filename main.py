import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.data_loader import load_and_preprocess_data, KeystrokeDataset
from src.model import BiEncoderModel
from src.train import train_model, TripletLoss
from src.evaluate import evaluate_model

# Main execution
if __name__ == '__main__':
    # Set the folder path
    folder_path = '/Users/irentala/PycharmProjects/fake-profile-detection-transformer/data'

    # Load dataset
    data, labels = load_and_preprocess_data(folder_path)

    # Prepare dataset and dataloader
    dataset = KeystrokeDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available, otherwise CPU
    model = BiEncoderModel(input_dim=data.shape[-1], model_dim=128, num_heads=8, num_layers=6).to(device)  # Initialize the model
    criterion = TripletLoss(margin=1.0)  # Initialize the triplet loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Initialize the optimizer

    # Cross-validation setup
    k_folds = 2
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Define number of epochs
    num_epochs = 10

    # Cross-validation training and evaluation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}')

        # Split data into training and validation sets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)

        # Initialize model, criterion, optimizer
        model = BiEncoderModel(input_dim=data.shape[-1], model_dim=128, num_heads=8, num_layers=6).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training the model
        for epoch in range(num_epochs):
            loss = train_model(model, train_loader, optimizer, criterion, device)  # Train the model for one epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")  # Print the loss for the current epoch

        print("Training complete for fold.")

        # Evaluate the model
        known_genuine_embeddings = []
        model.eval()
        with torch.no_grad():
            for data, labels in train_loader:
                data = data.to(device).float()
                embeddings_a, _ = model(data, data)
                known_genuine_embeddings.extend(embeddings_a)

        threshold = 0.5  # Example threshold, adjust based on your data and requirements
        accuracy = evaluate_model(model, known_genuine_embeddings, val_loader, threshold, device)
        print(f"Evaluation accuracy for fold {fold + 1}: {accuracy:.4f}")

    print("Cross-validation complete.")
