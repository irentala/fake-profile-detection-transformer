# Keystroke Dynamics with Transformer Model

This repository contains code for a keystroke dynamics project using a Transformer-based model and triplet loss.

## Directory Structure

- `src/`: Source code
  - `data_loader.py`: Functions for data loading and preprocessing.
  - `model.py`: Definition of the Transformer model and triplet loss.
  - `train.py`: Training loop.
  - `evaluate.py`: Evaluation function.
  - 'keystroke_analysis_tool.py' : This script processes raw keystroke logging data, 
     ensuring its integrity, cleaning the data, and generating enhanced insights such as 
     session IDs and key press/release times. 
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
5. To run keystroke_analysis_tool.py script
   ```sh
   python3 keystroke_analysis_tool.py [Data File Path]
    ```

## License

This project is licensed under the MIT License.
