import os
import pandas as pd
import sys
import re


def remove_invalid_keystrokes(df):
    return df.loc[df["Key"] != "<0>"]


def read_new_file(path: str):
    print(f"Attempting to read file: {path}")

    try:
        # Read the file assuming there are no headers
        df = pd.read_csv(path, header=None)

        # Assign meaningful column names manually
        df.columns = ["Direction", "Key", "Timestamp"]
        print("Assigned Columns:", df.columns)

    except FileNotFoundError:
        print(f"File not found: {path}")
        return None
    except ValueError as e:
        print(f"ValueError: {e}")
        return None

    df["Direction"] = df["Direction"].replace({0: "P", 1: "R"})
    df = df.astype({"Direction": str, "Key": str, "Timestamp": float})
    return remove_invalid_keystrokes(df)


def check_data_integrity(df):
    issues = []

    if df is None:
        return "Error: DataFrame is None, possibly due to a file read error."

    # Check 1: Ensure 'Direction' column only contains 'P' or 'R'
    if not df['Direction'].isin(['P', 'R']).all():
        issues.append("Error: 'Direction' column contains values other than 'P' or 'R'.")

    # Check 2: Ensure 'Timestamp' is numeric and positive
    if not pd.to_numeric(df['Timestamp'], errors='coerce').notnull().all():
        issues.append("Error: 'Timestamp' column contains non-numeric values.")
    if (df['Timestamp'] <= 0).any():
        issues.append("Error: 'Timestamp' column contains non-positive values.")

    # Check 3: Ensure timestamps are in increasing order
    if not df['Timestamp'].is_monotonic_increasing:
        issues.append("Error: Timestamps are not in strictly increasing order.")

    # Report the issues
    if not issues:
        return "Data Integrity Check Passed: No issues found."
    else:
        return "Data Integrity Check Failed with the following issues:\n" + "\n".join(issues)


# Function to extract user_ids from the filename
def extract_user_id(filename: str) -> str:
    base_name = os.path.basename(filename)
    user_id = base_name.split('_')[1]  # Assumes user ID is the second part of the filename
    return user_id


# Function to extract platform_ids based on the first character of the filename
def extract_platform_id(filename: str) -> int:
    first_char = os.path.basename(filename)[0].lower()
    if first_char == 'f':
        return 1
    elif first_char == 'i':
        return 2
    elif first_char == 't':
        return 3
    else:
        return 0  # Return 0 if none of the conditions match (optional)


def clean_key_column(df):
    # Convert the Key column to string to avoid any data type issues
    df['Key'] = df['Key'].astype(str)

    # Remove any whitespace and normalize Unicode characters
    df['Key'] = df['Key'].apply(lambda x: re.sub(r'\s+', '', x))
    df['Key'] = df['Key'].str.normalize('NFKC')  # Normalize the Unicode representation

    return df

def add_session_ids(df):
    session_id = 1
    x_pair_count = 0
    y_pair_count = 0
    last_key = None

    # Initialize session_id column
    df['session_id'] = session_id

    i = 0
    while i < len(df) - 1:
        # Clean the Key and Direction values by stripping extra spaces and normalizing
        key_1 = df.loc[i, 'Key'].strip().lower().replace("'", "")
        key_2 = df.loc[i + 1, 'Key'].strip().lower().replace("'", "")
        direction_1 = df.loc[i, 'Direction'].strip().upper()
        direction_2 = df.loc[i + 1, 'Direction'].strip().upper()

        # Debugging: Show the cleaned values being compared
        print("Bef IF", key_1[0], direction_1[0])
        # Check for x pair: P x followed by R x
        if key_1 == 'x' and direction_1 == 'P' and key_2 == 'x' and direction_2 == 'R':
            print("inside first IF", key_1[0], direction_1[0])
            if last_key == 'x':
                x_pair_count += 1
            else:
                x_pair_count = 1  # Start counting new sequence of 'x'
                last_key = 'x'
            i += 2  # Skip the next row since it's part of the pair
        # Check for y pair: P y followed by R y
        elif key_1 == 'y' and direction_1 == 'P' and key_2 == 'y' and direction_2 == 'R':
            print("inside else IF", key_1[0], direction_1[0])
            if last_key == 'y':
                y_pair_count += 1
            else:
                y_pair_count = 1  # Start counting new sequence of 'y'
                last_key = 'y'
            i += 2  # Skip the next row since it's part of the pair
        else:
            i += 1  # Move to the next row if no match

        # Increment session_id when both 5 pairs of x and y are found
        if x_pair_count == 5 and y_pair_count == 5:
            session_id += 1
            print(f"Incrementing session_id to {session_id} at row {i}")
            x_pair_count = 0  # Reset x count after session increment
            y_pair_count = 0  # Reset y count after session increment
            last_key = None  # Reset to allow new sequence

        # Set the session ID for the current row
        df.loc[i, 'session_id'] = session_id

    # For rows that were not covered in the loop, set the session ID
    df['session_id'] = df['session_id'].ffill().bfill()

    return df


def calculate_press_release_times(df):
    # Create new columns for press and release times
    df['press_time'] = None
    df['release_time'] = None

    # Dictionary to track the latest press times by key
    key_press_times = {}

    for index, row in df.iterrows():
        key = row['Key']
        timestamp = row['Timestamp']

        if row['Direction'] == 'P':  # If it's a press event
            key_press_times[key] = timestamp  # Track the press time
            df.at[index, 'press_time'] = timestamp  # Store the press time

        elif row['Direction'] == 'R':  # If it's a release event
            # Release time is directly the timestamp of the release event
            if key in key_press_times:
                df.at[index, 'press_time'] = key_press_times[key]  # Use the last recorded press time
                df.at[index, 'release_time'] = timestamp  # Set the release time
                del key_press_times[key]  # Remove the key as it is now processed
            else:
                # Handle the case where a release event is found without a prior press
                df.at[index, 'release_time'] = timestamp  # Store release time, but no corresponding press

    return df


# Function to process and save a single file
def process_file(input_file, output_folder):
    try:
        df = read_new_file(input_file)

        # Perform data integrity check
        integrity_result = check_data_integrity(df)
        print(integrity_result)

        if "Data Integrity Check Failed" in integrity_result:
            print(f"Skipping file {input_file} due to data integrity issues.")
            return None

        # Extract user_id and platform_id from the filename
        user_id = extract_user_id(input_file)
        platform_id = extract_platform_id(input_file)

        # Add the user_ids and platform_ids columns
        df["user_ids"] = user_id
        df["platform_ids"] = platform_id

        # Add session_id column based on 2 consecutive pairs of 'x' in 'Key' column
        df = add_session_ids(df)

        # Calculate press and release times
        df = calculate_press_release_times(df)

        # Reorder columns and filter rows with non-null press_time and release_time
        df = df[['Key', 'press_time', 'release_time', 'platform_ids', 'session_id', 'user_ids']]
        df = df.dropna(subset=['press_time', 'release_time'])

        # Define output file name and save the individual cleansed file
        output_file = os.path.join(output_folder, f"cleansed_{os.path.basename(input_file)}")
        df.to_csv(output_file, index=False)
        print(f"Cleansed data has been saved to {output_file}")

        return df
    except Exception as e:
        print(f"Failed to process {input_file}: {e}")
        return None


# Main execution for processing multiple files and saving consolidated output
def process_multiple_files(file_list):
    # Create a folder to store cleansed files
    output_folder = os.path.join(os.getcwd(), "cleansed")
    os.makedirs(output_folder, exist_ok=True)

    # DataFrame to hold consolidated data
    consolidated_df = pd.DataFrame()

    for input_file in file_list:
        print(f"Processing file: {input_file}")
        df = process_file(input_file, output_folder)

        if df is not None:
            consolidated_df = pd.concat([consolidated_df, df], ignore_index=True)

    # Save the consolidated data to a single file
    consolidated_output_file = os.path.join(output_folder, "cleansed_data.csv")
    consolidated_df.to_csv(consolidated_output_file, index=False)
    print(f"Consolidated data has been saved to {consolidated_output_file}")


if __name__ == "__main__":
    # Get the folder path from command line arguments
    folder_path = sys.argv[1] if len(sys.argv) > 1 else None

    if folder_path and os.path.isdir(folder_path):
        # Process all CSV files in the folder
        file_list = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                     filename.endswith(".csv")]
        process_multiple_files(file_list)
    else:
        print("Error: Please provide a valid folder path containing CSV files.")
