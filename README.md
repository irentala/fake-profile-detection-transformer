# Keystroke Dynamics with Transformer Model

This repository contains code for a keystroke dynamics project using a Transformer-based model and triplet loss.

## Directory Structure

- `src/`: Source code
  - `data_loader.py`: Functions for data loading and preprocessing.
  - `model.py`: Definition of the Transformer model and triplet loss.
  - `train.py`: Training loop.
  - `evaluate.py`: Evaluation function.
- `data/`: Directory for storing keystroke data files.
- `main.py`: Main script for running training and evaluation.
- `requirements.txt`: List of dependencies.

## How to Run

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/keystroke-dynamics-transformer.git
    cd keystroke-dynamics-transformer
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Place your keystroke data files in the `data/` directory.

4. Run the main script:
    ```sh
    python main.py
    ```

## License

This project is licensed under the MIT License.
