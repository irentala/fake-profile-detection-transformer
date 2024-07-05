# src/evaluate.py

import torch  # PyTorch library for tensor operations and neural networks

# Evaluation function
def evaluate_model(model, known_genuine_embeddings, test_dataloader, threshold, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Initialize the correct count
    total = 0  # Initialize the total count
    with torch.no_grad():  # Disable gradient computation
        for data, labels in test_dataloader:  # Iterate over batches of test data
            data, labels = data.to(device), labels.to(device)  # Move data and labels to the specified device
            data = data.float()  # Convert data to float32
            embeddings = model(data)  # Compute embeddings for test sequences
            for embedding in embeddings:  # Iterate over embeddings
                distances = [torch.nn.functional.pairwise_distance(embedding.unsqueeze(0), genuine_embedding.unsqueeze(0)).item() for genuine_embedding in known_genuine_embeddings]  # Compute distances to known genuine embeddings
                min_distance = min(distances)  # Get the minimum distance
                print(f"Distances: {distances}, Min Distance: {min_distance}")  # Print the distances for debugging
                if min_distance < threshold:  # Classify as genuine if distance is below the threshold
                    correct += 1
                total += 1
    accuracy = correct / total  # Compute the accuracy
    print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy:.4f}")  # Print the accuracy
    return accuracy  # Return the accuracy
