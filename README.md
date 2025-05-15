# 🎓 PROG2900 - Bachelor Thesis Project #
Title: Cursor Dynamics for Deception Detection <br />
Author: Nemanja Tosic <br />
Supervisor: Kiran Raja <br />
Company: Mobai AS <br />

Norwegian University of Science and Technology <br />
Department of Computer Science <br />
Gjøvik, Spring 2025 <br />

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)


## Project Overview

This repository contains all data, scripts, and code associated with bachelor thesis titled *"Cursor Dynamics for Deception Detection"*. The study explores whether deceptive intent can be detected from subtle variations in mouse movement behavior during binary question answering. Two modeling approaches were investigated: a classical machine learning pipeline using Random Forest, and a deep learning architecture combining LSTM, GRU, and attention mechanisms.

A total of 700 mouse movement samples were collected in a controlled experiment. Each sample was labeled as either **truthful** or **deceitful**, and analyzed using dynamic features such as velocity, acceleration, jerk, curvature, hesitation, pauses, and total movement time.

## Repository Structure

The Repository Structure section outlines the organization of all project files, including scripts for classical and deep learning models, raw mouse movement data, visualization tools, and the web-based data collection system.

```
PROG2900-Bachelor-project-2025/
├── classical_model_training/                      # Classical machine learning pipeline (Random Forest)
│   ├── classical_graph/                           # Visualizations specific to classical model results
│   │   └── ...                                     # (e.g., confusion matrix, feature importance plots)
│   ├── model_training_rf_cv_fold.py               # Random Forest model with Stratified K-Fold CV
│   └── model_training_rf_v1.py                    # Baseline Random Forest model
│
├── data/                                          # Raw mouse movement data organized by class
│   ├── deceitful/                                 # 350 JSON samples representing deceptive responses
│   │   └── ...                                     # Individual participant trials (JSON format)
│   └── truthful/                                  # 350 JSON samples representing truthful responses
│       └── ...                                     # Same structure as above
│
├── data_analysys_stats/                           # Scripts and data for exploratory data analysis and statistics
│   ├── averaged_data/                             # Precomputed average trajectories and stats per class
│   │   ├── deceitful_averaged_result_interpolated.json      # Interpolated average trajectory for deceitful
│   │   ├── deceitful_mouse_stats_summary.json               # Summary stats for deceitful mouse behavior
│   │   ├── truthful_averaged_result_interpolated.json       # Interpolated average trajectory for truthful
│   │   └── truthful_mouse_stats_summary.json                # Summary stats for truthful mouse behavior
│   ├── graph_charts/                              # Exported visual plots
│   │   └── ...                                     # Graphs for speed, acceleration, jerk, etc.
│   └── utils/                                     # Python scripts for generating stats and charts
│       ├── acj_interpolation_plot.py              # Generates acceleration/curvature/jerk trajectory plots
│       ├── acj_plot_graphs.py                     # Draws values as separate time-series plots
│       ├── average_mouse_pattern_graphs.py        # Plots average paths for each class
│       ├── average_mouse_stats_chart.py           # Bar charts of avg. movement features per class
│       ├── average_mouse_stats.py                 # Calculates summary statistics from JSON
│       ├── jspk_plot_graphs.py                    # Plots jerk spike patterns
│       └── labeling_data_training.py              # Helper script for tagging samples with class labels
│
├── deep_learning_model_training/                  # Deep learning model training (LSTM + GRU + Attention)
│   ├── best_dl/                                   # Saved best-performing model version and metadata
│   │   └── ...                                     # Could include architecture visualizations, checkpoints
│   ├── export_training_log_table.py              # Exports all training metrics (acc, F1, etc.) into a CSV
│   ├── model_fold_1.h5                            # Trained model for Fold 1 (Keras HDF5 format)
│   ├── model_fold_2.h5                            # Trained model for Fold 2
│   ├── model_fold_3.h5                            # Trained model for Fold 3
│   ├── model_fold_4.h5                            # Trained model for Fold 4
│   ├── model_fold_5.h5                            # Trained model for Fold 5
│   ├── model_training_lstm_gru_v1.py             # Version 1: Deep model training script with fixed setup
│   ├── model_training_lstm_gru_v2.py             # Version 2: More advanced DL training (e.g., attention, callbacks)
│   ├── training_log_fold_1.csv                   # Training history: accuracy, F1, loss (Fold 1)
│   ├── training_log_fold_2.csv
│   ├── training_log_fold_3.csv
│   ├── training_log_fold_4.csv
│   └── training_log_fold_5.csv
│
├── public/                                       # Web interface frontend for data collection (browser)
│   ├── index.html                                 # Static HTML page with question interface
│   ├── script.js                                  # JavaScript for mouse tracking and interaction
│   └── styles.css                                 # Basic CSS styling for the UI
│
├── package-lock.json                             # Lock file that pins exact versions of npm packages
├── package.json                                  # Defines Node.js dependencies (used for server + frontend)
├── README.md                                     # Project documentation and usage instructions
├── requirements.txt                              # Python dependencies used across scripts and models
└── server.js                                     # Node.js server for capturing and storing mouse data (JSON)

```

## Installation

### 1. Clone repository:
```
git clone https://github.com/nemanjant/PROG2900-Bachelor-project-2025.git
cd PROG2900-Bachelor-project-2025
```

### 2. Install dependencies:
Ensure you have Node.js and Python 3.11 installed. Then, run:
```
npm install
pip install -r requirements.txt
```

## Usage Instructions

### 1. Collecting Data (Optional)

Run the Node.js backend. The backend saves mouse data to JSON files locally during the experiment. After collection, data needs to be sorted in two folders data/truthful and data/deceitful. Samples needs to be labeled.
```
node server.js
```




