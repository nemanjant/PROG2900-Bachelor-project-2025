import json
import glob
import os

# Paths to the folders containing the JSON response files
truthful_folder = "data/truthful/"
deceptive_folder = "data/deceitful/"

# Label to assign 0 for truthful, 1 for deceptive
def add_labels_to_json_files(folder, label):
    # Glob all .json files in the specified folder
    file_pattern = os.path.join(folder, "*.json")
    files = glob.glob(file_pattern)

    # Process each JSON file found
    for file_path in files:
        # Load the existing JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Add or update the 'label' field
        data["label"] = label

        # Save the updated JSON back to the same file, with pretty formatting
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        # Print a message indicating successful labeling
        print(f"Added label {label} to {file_path}")


if __name__ == "__main__":
    # Add label 0 for truthful responses
    add_labels_to_json_files(truthful_folder, 0)
    # Add label 1 for deceptive responses
    add_labels_to_json_files(deceptive_folder, 1)

