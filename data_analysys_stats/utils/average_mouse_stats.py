import os
import json

# === Mouse Statistics Summaries Generator ===
# This script processes raw mouse movement JSON data,
# computes key summary statistics for two datasets ('truthful' and 'deceitful'),
# and saves the results as JSON files in an output directory.


def load_json_files(folder_path):
    """
    Load all JSON files from a specified folder.

    Args:
        folder_path (str): Path to the directory containing .json files.

    Returns:
        list: A list of dictionaries loaded from each JSON file.
    """
    data = []
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Only process files with .json extension
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                # Parse the JSON content and append to list
                data.append(json.load(f))
    return data


def compute_summary_stats(data):
    """
    Compute average statistics across all data entries.

    Metrics calculated:
      - averageTotalTime: mean of totalTime values
      - averageSpeed: mean of averageSpeed values
      - averageJerkSpikeCount: mean count of jerk spikes
      - averageHesitation: mean hesitation measure
      - averagePauseDuration: mean duration of all pauses
      - averagePauseCount: mean number of pause points per entry

    Args:
        data (list): List of dictionaries, each representing one session's data.

    Returns:
        dict: A dictionary of computed average statistics.
    """
    total_times = []
    speeds = []
    jerks = []
    hesitations = []
    pause_counts = []
    pause_durations = []

    # Aggregate metrics from each entry
    for entry in data:
        total_times.append(entry.get('totalTime', 0))
        speeds.append(entry.get('averageSpeed', 0))
        jerks.append(entry.get('jerkSpikeCount', 0))
        hesitations.append(entry.get('hesitation', 0))

        pauses = entry.get('pausePoints', [])
        pause_counts.append(len(pauses))
        # Collect each pause duration for averaging
        for p in pauses:
            pause_durations.append(p.get('duration', 0))

    def safe_avg(arr):
        """
        Compute the average of a list, returning 0 for empty lists.
        """
        return sum(arr) / len(arr) if arr else 0

    # Assemble the summary dictionary
    return {
        'averageTotalTime': safe_avg(total_times),
        'averageSpeed': safe_avg(speeds),
        'averageJerkSpikeCount': safe_avg(jerks),
        'averageHesitation': safe_avg(hesitations),
        'averagePauseDuration': safe_avg(pause_durations),
        'averagePauseCount': safe_avg(pause_counts)
    }


def save_stats_to_json(stats, output_path):
    """
    Save statistics dictionary to a file in JSON format.

    Args:
        stats (dict): Summary statistics to save.
        output_path (str): Path to the output JSON file.
    """
    # Write the stats dict to a JSON file with indentation for readability
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)


def main():
    """
    Main execution function:
      1. Prepare input and output directories.
      2. Load data for each dataset label.
      3. Compute summary statistics.
      4. Save summaries to JSON files.
    """
    base_dir = 'data'
    output_dir = 'averaged_data'

    # Ensure the output directory exists, create if missing
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Define the two datasets to process
    datasets = {
        'truthful': os.path.join(base_dir, 'truthful'),
        'deceitful': os.path.join(base_dir, 'deceitful')
    }

    # Loop over each label/folder pair
    for label, folder in datasets.items():
        if not os.path.isdir(folder):
            print(f"Warning: directory '{folder}' not found, skipping.")
            continue

        print(f"Processing '{label}' data from '{folder}'...")
        # Load JSON session files
        data = load_json_files(folder)
        # Compute the summary stats
        stats = compute_summary_stats(data)

        # Construct the output filename and save
        output_file = os.path.join(output_dir, f"{label}_mouse_stats_summary.json")
        save_stats_to_json(stats, output_file)
        print(f"Saved summary to '{output_file}'")


if __name__ == '__main__':
    # Entry point of the script
    main()
