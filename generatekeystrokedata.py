import pandas as pd
import numpy as np

# Number of participants
num_participants = 5
# Number of keystrokes per participant
num_keystrokes = 100

# Initialize the data dictionary
data = {
    'PARTICIPANT_ID': [],
    'PRESS_TIME': [],
    'RELEASE_TIME': []
}

# Generate synthetic data
for participant_id in range(num_participants):
    # Generate random press times as a cumulative sum to simulate typing
    press_times = np.cumsum(np.random.rand(num_keystrokes) * 100)
    # Generate random release times that are always after the press times
    release_times = press_times + np.random.rand(num_keystrokes) * 10
    # Create a list of participant IDs
    participant_ids = [participant_id] * num_keystrokes

    # Append generated data to the dictionary
    data['PARTICIPANT_ID'].extend(participant_ids)
    data['PRESS_TIME'].extend(press_times)
    data['RELEASE_TIME'].extend(release_times)

# Create a DataFrame from the data dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a text file with tab-separated values
file_path = 'synthetic_keystrokes.txt'
df.to_csv(file_path, sep='\t', index=False)

print(f"Synthetic dataset saved to {file_path}")
