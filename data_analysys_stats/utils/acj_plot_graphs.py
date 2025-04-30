import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# === Loaders ===
def load_array(filepath, key):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get(key, [])

def load_mouse_movements(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return np.array(data.get("mouseMovements", []))

# === Plotters ===
def plot_series_comparison(series_lie, series_truth, title, ylabel, filename, sigma=2):
    min_len = min(len(series_lie), len(series_truth))
    x = np.arange(min_len)
    series_lie = gaussian_filter1d(series_lie[:min_len], sigma=sigma)
    series_truth = gaussian_filter1d(series_truth[:min_len], sigma=sigma)

    plt.figure(figsize=(12, 5))
    plt.plot(x, series_lie, label="Deceitful", color="red", alpha=0.8)
    plt.plot(x, series_truth, label="Truthful", color="blue", alpha=0.8)
    plt.title(f"{title} Over Time")
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_mouse_path_comparison(path_lie, path_truth, filename):
    min_len = min(len(path_lie), len(path_truth))
    path_lie = np.array(path_lie[:min_len])
    path_truth = np.array(path_truth[:min_len])

    plt.figure(figsize=(8, 6))
    plt.plot(path_lie[:, 0], path_lie[:, 1], label="Deceitful", color="red", alpha=0.8)
    plt.plot(path_truth[:, 0], path_truth[:, 1], label="Truthful", color="blue", alpha=0.8)
    plt.title("Mouse Movement Path Comparison: Truthful vs Deceitful")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# === Main ===
if __name__ == "__main__":
    lie_file = "averaged_data/deceitful_averaged_result_interpolated.json"
    truth_file = "averaged_data/truthful_averaged_result_interpolated.json"

    # Acceleration
    lie_acc = load_array(lie_file, "accelerations")
    truth_acc = load_array(truth_file, "accelerations")
    plot_series_comparison(lie_acc, truth_acc, "Acceleration", "Acceleration", "graph_charts/acceleration_comparison.png")

    # Curvature
    lie_curv = load_array(lie_file, "curvatures")
    truth_curv = load_array(truth_file, "curvatures")
    plot_series_comparison(lie_curv, truth_curv, "Curvature", "Curvature", "graph_charts/curvature_comparison.png")

    # Jerk
    lie_jerk = load_array(lie_file, "jerks")
    truth_jerk = load_array(truth_file, "jerks")
    plot_series_comparison(lie_jerk, truth_jerk, "Jerk", "Jerk", "graph_charts/jerk_comparison.png")
