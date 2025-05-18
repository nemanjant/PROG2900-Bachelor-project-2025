import os
import json
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("default")


def smooth(data, window_size=3):

    # Create an averaging window and convolve with the data
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def plot_jerk_curve(jerks, label, line_color, output_path):
    # Smooth the raw jerk signal
    smoothed = smooth(jerks)

    # Dynamic threshold based on multiple criteria:
    # 1) Four times the mean absolute jerk (t_mean)
    # 2) 95th percentile of absolute jerk (t_pct95)
    # 3) Fixed cap at 1e6 (t_fixed)
    # 4) Lower bound of 400,000 to avoid too low thresholds
    mean_abs = np.mean(np.abs(jerks))
    t_mean = mean_abs * 4
    t_pct95 = np.percentile(np.abs(jerks), 95)
    t_fixed = 20000
    # Combine criteria: pick a threshold that isn't too low or too high
    threshold = min(max(t_mean, 400000), 1e6, t_pct95, t_fixed)

    # Identify indices where jerk magnitude exceeds the threshold
    spikes = [i for i, v in enumerate(jerks) if abs(v) >= threshold]
    # Map spike indices to smoothed values for plotting
    spike_vals = [smoothed[i] for i in spikes]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    # Plot the smoothed jerk curve
    ax.plot(smoothed, color=line_color, linewidth=2, label="Jerk (smoothed)")

    # Draw a horizontal threshold line for reference
    ax.axhline(threshold, color='gray', ls='--', lw=1, label="Threshold")

    # If there are any spikes, mark them and draw vertical lines
    if spikes:
        ax.scatter(spikes, spike_vals, color='gray', s=30, label=f"Spikes ({len(spikes)})", zorder=5)
        for idx in spikes:
            ax.axvline(idx, color='gray', ls=':', alpha=0.3)

    # Set titles and axis labels for clarity
    ax.set_title(f"{label.capitalize()}: Jerk Spikes Over Time", fontsize=14)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Jerk", fontsize=12)
    ax.grid(True, ls='--', alpha=0.3)
    ax.legend(frameon=False, fontsize=11)
    plt.tight_layout()

    # Ensure output directory exists, then save and close the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main():
    # Define input files and line colors per label
    inputs = {
        "truthful": ("data_analysys_stats/averaged_data/truthful_averaged_result_interpolated.json", "blue"),
        "deceitful": ("data_analysys_stats/averaged_data/deceitful_averaged_result_interpolated.json", "red"),
    }
    out_dir = "data_analysys_stats/graph_charts"
    os.makedirs(out_dir, exist_ok=True)

    # Iterate through each label to process its data
    for label, (path, color) in inputs.items():
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue

        jerks = data.get("jerks", [])
        if not jerks:
            print(f"No 'jerks' data in {path}")
            continue

        # Construct output filename and plot
        filename = f"{label}_jerk_spikes.png"
        output_path = os.path.join(out_dir, filename)
        plot_jerk_curve(jerks, label, line_color=color, output_path=output_path)
        print(f"Saved {label} jerk plot to {output_path}")


if __name__ == "__main__":
    main()




