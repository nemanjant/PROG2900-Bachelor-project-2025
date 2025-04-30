import os
import json
import numpy as np
from glob import glob
from scipy.interpolate import CubicSpline

# === Settings ===
INTERPOLATION_POINTS = 100

# === Helper Functions ===

def interpolate_to_fixed_length(array, target_len=INTERPOLATION_POINTS):
    if not array:
        return []

    # Convert dicts {"x":..., "y":...} to [x, y]
    if isinstance(array[0], dict) and "x" in array[0] and "y" in array[0]:
        array = [[pt["x"], pt["y"]] for pt in array]

    if len(array) == 1:
        # replicate the single point
        return [array[0] for _ in range(target_len)]

    x_old = np.linspace(0, 1, len(array))
    x_new = np.linspace(0, 1, target_len)

    # 2D coordinate list
    if isinstance(array[0], list) and len(array[0]) == 2:
        xs = [pt[0] for pt in array]
        ys = [pt[1] for pt in array]
        cs_x = CubicSpline(x_old, xs)
        cs_y = CubicSpline(x_old, ys)
        return [[float(x), float(y)] for x, y in zip(cs_x(x_new), cs_y(x_new))]

    # 1D values
    cs = CubicSpline(x_old, array)
    return cs(x_new).tolist()

def average_interpolated(arrays, target_len=INTERPOLATION_POINTS):
    clean = []
    for arr in arrays:
        try:
            interp = interpolate_to_fixed_length(arr, target_len)
            # Ensure correct shape: either a list of 2-tuples or a list of scalars
            if interp and isinstance(interp[0], list) and len(interp[0]) == 2:
                clean.append(interp)
            elif interp and isinstance(interp[0], (int, float)):
                clean.append(interp)
        except Exception:
            continue

    if not clean:
        return []

    return np.mean(clean, axis=0).tolist()

def normalize_jerks(jerks):
    abs_mean = np.mean(np.abs(jerks))
    if abs_mean == 0:
        return jerks
    return [j/abs_mean for j in jerks]

# === Core averaging per-folder ===

def average_json_from_folder(folder_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    json_files = glob(os.path.join(folder_path, "*.json"))
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return

    # fields to aggregate
    numeric_fields = ["accelerations", "curvatures", "timestamps"]
    array_fields   = ["mouseMovements", "pausePoints"]
    scalar_fields  = ["totalTime", "averageSpeed", "jerkSpikeCount", "hesitation"]

    buckets_num   = {f: [] for f in numeric_fields}
    buckets_arr   = {f: [] for f in array_fields}
    buckets_sclr  = {f: [] for f in scalar_fields}

    for fp in json_files:
        with open(fp, 'r') as f:
            data = json.load(f)
        for f in numeric_fields:
            if f in data: buckets_num[f].append(data[f])
        for f in array_fields:
            if f in data: buckets_arr[f].append(data[f])
        for f in scalar_fields:
            if f in data: buckets_sclr[f].append(data[f])

    result = {
        "question": f"Average over {len(json_files)} samples",
        "answer": "N/A"
    }

    # interpolate & average lists
    for f in numeric_fields:
        result[f] = average_interpolated(buckets_num[f])
    for f in array_fields:
        result[f] = average_interpolated(buckets_arr[f])
    # simple mean of scalars
    for f in scalar_fields:
        vals = buckets_sclr[f]
        result[f] = float(np.mean(vals)) if vals else 0.0

    # compute jerks and normalized jerks
    if result.get("accelerations"):
        jerks = np.diff(result["accelerations"]).tolist()
        # round off near-zero noise
        jerks = [round(j, 10) if abs(j)>0 else 0 for j in jerks]
        result["jerks"]           = jerks
        result["jerksNormalized"] = normalize_jerks(jerks)

    # write
    with open(output_path, 'w') as out:
        json.dump(result, out, indent=2)

    print(f"→ saved: {output_path}")

# === Entry point ===

if __name__ == "__main__":
    # ensure averaged_data folder exists
    os.makedirs("averaged_data", exist_ok=True)

    # process both truthful and deceitful
    average_json_from_folder(
        folder_path="data/truthful",
        output_path="averaged_data/truthful_averaged_result_interpolated.json"
    )
    average_json_from_folder(
        folder_path="data/deceitful",
        output_path="averaged_data/deceitful_averaged_result_interpolated.json"
    )

