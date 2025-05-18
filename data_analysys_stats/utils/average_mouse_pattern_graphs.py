import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection


# Data loading 
def load_mouse_movements_from_json(folder_path):
    all_movements = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), 'r') as f:
                data = json.load(f)
                all_movements.append(data["mouseMovements"])
    return all_movements

def interpolate_movements(movements, num_points=100):
    interpolated = []
    for path in movements:
        path = np.array(path)
        if len(path) < 2:
            continue
        t = np.linspace(0, 1, len(path))
        new_t = np.linspace(0, 1, num_points)
        kind = 'cubic' if len(path) >= 4 else 'linear'
        interp_x = interp1d(t, path[:, 0], kind=kind)
        interp_y = interp1d(t, path[:, 1], kind=kind)
        interpolated.append(np.stack((interp_x(new_t), interp_y(new_t)), axis=-1))
    return np.array(interpolated)

def average_mouse_movements(interpolated_paths):
    return np.mean(interpolated_paths, axis=0)

# Enhanced plotting
def plot_single_path(avg_array, label, color, output_file):
    x, y = avg_array[:, 0], avg_array[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    # light background
    ax.set_facecolor('#f7f7f7')
    fig.patch.set_facecolor('white')

    # draw the main path
    ax.plot(x, y, color=color, linewidth=3, label=f"{label} Path")

    # dashed grey line from start→end
    ax.plot(
        [x[0], x[-1]], [y[0], y[-1]],
        linestyle='--', color='gray', linewidth=1.5,
        label="Start → End"
    )

    # small grey start/end markers
    ax.scatter(x[0], y[0], color='gray', s=40, marker='o')
    ax.scatter(x[-1], y[-1], color='gray', s=40, marker='o')

    # annotate points
    ax.annotate("Start", (x[0], y[0]), xytext=(8, -8),
                textcoords='offset points', color='gray', fontsize=10)
    ax.annotate("End", (x[-1], y[-1]), xytext=(8, -8),
                textcoords='offset points', color='gray', fontsize=10)

    # styling
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.tick_params(direction='out', length=6, width=1.2)

    # grid, labels, legend
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_title(f"{label} – Average Mouse Movement", fontsize=14)
    ax.invert_yaxis()  # match screen coords
    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

# Data Load
if __name__ == "__main__":
    truthful_path  = "data/truthful"
    deceitful_path = "data/deceitful"

    truthful_movements  = load_mouse_movements_from_json(truthful_path)
    deceitful_movements = load_mouse_movements_from_json(deceitful_path)

    truthful_interp  = interpolate_movements(truthful_movements, num_points=100)
    deceitful_interp = interpolate_movements(deceitful_movements, num_points=100)

    truth_avg = average_mouse_movements(truthful_interp)
    lie_avg   = average_mouse_movements(deceitful_interp)

    plot_single_path(truth_avg, label="Truthful", color="blue", output_file="data_analysys_stats/graph_charts/truthful_average_mouse_path.png")
    plot_single_path(lie_avg,   label="Deceitful",   color="red", output_file="data_analysys_stats/graph_charts/deceitful_average_mouse_path.png")
