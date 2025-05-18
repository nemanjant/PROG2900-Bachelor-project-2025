import json
import matplotlib.pyplot as plt
import numpy as np

# Load the summary metrics from a JSON file
def load_summary_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Plot all metrics except average speed
def plot_metrics_comparison(lie_data, truth_data):
    metric_labels = [
        "Total Time (s)",
        "Jerk Spikes",
        "Hesitation",
        "Pause Duration",
        "Pause Count"
    ]
    keys = [
        "averageTotalTime",
        "averageJerkSpikeCount",
        "averageHesitation",
        "averagePauseDuration",
        "averagePauseCount"
    ]

    lie_values = [lie_data.get(k, 0) for k in keys]
    truth_values = [truth_data.get(k, 0) for k in keys]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, lie_values, width, label='Deceitful', color='red', alpha=0.9)
    bars2 = ax.bar(x + width/2, truth_values, width, label='Truthful', color='blue', alpha=0.9)

    ax.set_ylabel('Values')
    ax.set_title('Summary Statistics: Deceitful vs Truth')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("data_analysys_stats/graph_charts/average_mouse_stats_chart_metrics.png")
    plt.close()

# Plot average speed separately
def plot_speed_comparison(lie_data, truth_data):
    lie_speed = lie_data.get("averageSpeed", 0)
    truth_speed = truth_data.get("averageSpeed", 0)

    labels = ['Deceitful', 'Truthful']
    values = [lie_speed, truth_speed]

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(labels, values, color=['red', 'blue'], alpha=0.9)

    ax.set_ylabel('Average Speed (px/s)')
    ax.set_title('Average Speed Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("data_analysys_stats/graph_charts/average_mouse_speed_chart.png")
    plt.close()

# === USAGE ===
lie_file = "data_analysys_stats/averaged_data/deceitful_mouse_stats_summary.json"
truth_file = "data_analysys_stats/averaged_data/truthful_mouse_stats_summary.json"

lie_metrics = load_summary_metrics(lie_file)
truth_metrics = load_summary_metrics(truth_file)

plot_metrics_comparison(lie_metrics, truth_metrics)
plot_speed_comparison(lie_metrics, truth_metrics)
