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
├── classical_model_training/        # Python scripts for training, evaluating the Random Forest model.
│   ├── train_random_forest.py       # Baseline RF classifier with train/test split.
│   ├── train_rf_5fold.py            # Stratified 5-fold cross-validation version of the RF model.
│   └── utils_rf.py                  # Shared functions for feature loading and preprocessing.
│
├── deep_learning_model_training/    # Deep learning pipeline using LSTM → GRU → Attention.
│   ├── train_deep_model.py          # Main training script with 5-fold CV, callbacks, focal loss, etc.
│   ├── model_definition.py          # Model architecture (sequence + meta-features).
│   ├── data_loader.py               # Loads JSON, extracts features, sequences, and labels.
│   └── evaluation_utils.py          # Functions for confusion matrix, ROC, training curves, etc.
│
├── data/                            # Mouse movement dataset.
│   ├── truthful/                    # 350 JSON files representing truthful responses.
│   ├── deceitful/                   # 350 JSON files representing deceptive responses.
│   └── sample_schema.md             # Description of JSON file structure and parameters.
│
├── data_analysys_stats/            # Scripts for exploratory analysis and statistical plots.
│   ├── visualize_trajectories.py    # Plots average movement paths for truthful vs. deceitful.
│   ├── feature_stats_plot.py        # Generates histograms, bar plots, and time series.
│   ├── correlation_matrix.py        # Creates feature correlation heatmaps.
│   └── feature_importance_plot.py   # Visualizes feature importance (classical + DL models).
│
├── public/                          # Frontend of the web experiment interface.
│   ├── index.html                   # Main UI for question prompts and response buttons.
│   ├── style.css                    # Custom styling for interface.
│   └── script.js                    # JavaScript logic for cursor tracking and client-side events.
│
├── node_modules/                    # Auto-generated dependencies for the Node.js backend.
│
├── package.json                     # Metadata and dependencies for Node.js environment.
├── package-lock.json                # Exact dependency tree for reproducible installs.
│
├── server.js                        # Node.js backend for receiving and storing JSON mouse data.
│
└── README.md                        # Project documentation.
```
