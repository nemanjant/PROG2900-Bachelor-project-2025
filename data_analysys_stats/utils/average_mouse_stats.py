import os
import json


def load_json_files(folder_path):
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

# Compute average values for iven parameters
def compute_summary_stats(data):
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
    # Write the stats dict to a JSON file with indentation for readability
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)


def main():
    base_dir = 'data'
    output_dir = 'data_analysys_stats/averaged_data'

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
