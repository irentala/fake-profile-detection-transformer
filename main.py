# main.py

import torch  # PyTorch library for tensor operations and neural networks
from torch.utils.data import DataLoader  # PyTorch class for data loading
from sklearn.model_selection import KFold  # Library for K-Fold cross-validation
from src.data_loader import load_and_preprocess_data, KeystrokeDataset  # Import data loading functions and dataset class
from src.model import TransformerKeystrokeModel, TripletLoss  # Import model and loss function
from src.train import train_model  # Import training function
from src.evaluate import evaluate_model  # Import evaluation function

if __name__ == '__main__':
    folder_path = '/Users/irentala/PycharmProjects/fake-profile-detection-transformer/data'  # Set the folder path

    data, labels = load_and_preprocess_data(folder_path)  # Load and preprocess the data
    dataset = KeystrokeDataset(data, labels)  # Create the dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Create the dataloader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set the device to GPU if available, otherwise CPU
    model = TransformerKeystrokeModel(input_dim=data.shape[-1], model_dim=128, num_heads=8, num_layers=6).to(device)  # Initialize the model
    criterion = TripletLoss(margin=1.0)  # Initialize the triplet loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Initialize the optimizer

    k_folds = 5  # Number of folds for cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True)  # Create the KFold cross-validator
    num_epochs = 10  # Number of epochs for training

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):  # Iterate over each fold
        print(f'Fold {fold + 1}')  # Print the current fold

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)  # Create the training sampler
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)  # Create the validation sampler

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)  # Create the training dataloader
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)  # Create the validation dataloader

        model = TransformerKeystrokeModel(input_dim=data.shape[-1], model_dim=128, num_heads=8, num_layers=6).to(device)  # Initialize the model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Initialize the optimizer

        for epoch in range(num_epochs):  # Iterate over epochs
            loss = train_model(model, train_loader, optimizer, criterion, device)  # Train the model for one epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")  # Print the loss for the current epoch

        print("Training complete for fold.")  # Print that training is complete for the fold

        known_genuine_embeddings = []  # List to hold known genuine embeddings
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for data, labels in train_loader:  # Iterate over training data to get embeddings
                data = data.to(device).float()  # Move data to the device and convert to float32
                embeddings = model(data)  # Compute embeddings
                known_genuine_embeddings.extend(embeddings)  # Add embeddings to the list

        threshold = 0.5  # Example threshold
        accuracy = evaluate_model(model, known_genuine_embeddings, val_loader, threshold, device)  # Evaluate the model
        print(f"Evaluation accuracy for fold {fold + 1}: {accuracy:.4f}")  # Print the evaluation accuracy for the fold

    print("Cross-validation complete.")  # Print that cross-validation is complete
