# 🎓 PROG2900 - Bachelor Thesis Project

[![GitHub Repo Stars](https://img.shields.io/github/stars/nemanjant/PROG2900-Bachelor-project-2025?style=social)](https://github.com/nemanjant/PROG2900-Bachelor-project-2025)  
[![GitHub Forks](https://img.shields.io/github/forks/nemanjant/PROG2900-Bachelor-project-2025?style=social)](https://github.com/nemanjant/PROG2900-Bachelor-project-2025/fork)  
[![Last Commit](https://img.shields.io/github/last-commit/nemanjant/PROG2900-Bachelor-project-2025)](https://github.com/nemanjant/PROG2900-Bachelor-project-2025/commits/main)  
[![License](https://img.shields.io/badge/license-Academic%20Use-blue.svg)](LICENSE)

**Title:** Cursor Dynamics for Deception Detection  
**Author:** Nemanja Tosic  
**Supervisor:** Kiran Raja  
**Company:** Mobai AS  

Norwegian University of Science and Technology  
Department of Computer Science  
Gjøvik, Spring 2025

---

## Abstract

This project investigates whether deceptive intent can be inferred from subtle variations in mouse cursor dynamics during binary question answering. Two modeling strategies—a handcrafted-feature Random Forest and a hybrid LSTM-GRU-Attention network—are developed and compared to assess their effectiveness.

---

## Table of Contents

- [Background](#background)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Classical Model Architecture](#classical-model-architecture)
- [Deep Learning Architecture](#deep-learning-architecture)
- [Project Workflow](#project-workflow)
- [Results & Evaluation](#results--evaluation)
- [Conclusions](#conclusions)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [GitHub Stats](#github-stats)
- [Thesis](#thesis)

---

## Background

This repository hosts all code and data for the bachelor thesis *"Cursor Dynamics for Deception Detection"*. The study explores whether deceptive intent can be detected from mouse movement behavior, comparing a classical machine learning approach with a deep learning architecture.

---

## Repository Structure

<details>
<summary>Click to expand repository layout</summary>

```bash
PROG2900-Bachelor-project-2025/
├── classical_model_training/         # Classical Random Forest models
├── deep_learning_model_training/     # LSTM-GRU-Attention models and logs
├── data/                             # Raw mouse movement JSON files
├── data_analysys_stats/              # Visualization and statistical analysis scripts
├── public/                           # Frontend UI for data collection
├── server.js                         # Node.js backend for storing JSON data
├── package.json / lock               # Node.js config and dependencies
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation (you are here)
```

</details>

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/nemanjant/PROG2900-Bachelor-project-2025.git
cd PROG2900-Bachelor-project-2025
```

### 2. Install dependencies
Make sure you have Python 3.11 and Node.js installed:
```bash
npm install
pip install -r requirements.txt
```
---

## Usage

### 1. Data Collection
To run the local web experiment:
```bash
node server.js
```
Mouse movement responses will be saved to `data/` as structured JSON files. After collection, sort files into `data/truthful/` and `data/deceitful/`.

### 2. Data Analysis
Use the scripts in `data_analysys_stats/utils/` to process and visualize data:
```bash
# Generate summary statistics
python average_mouse_stats.py

# Interpolate average cursor paths
python average_mouse_pattern_graphs.py

# Plot behavioral features
python average_mouse_stats_chart.py
python acj_plot_graphs.py
python acj_interpolation_plot.py
python jspk_plot_graphs.py

# Add class labels for modeling
python labeling_data_training.py
```
Outputs are saved in `data_analysys_stats/averaged_data/` and `graph_charts/`.


### 3. Model Training

#### Classical Model (Random Forest)
- **Baseline (80/20 split):**
```bash
python classical_model_training/model_training_rf_v1.py
```

- **5-Fold Cross-Validation:**
```bash
python classical_model_training/model_training_rf_cv_fold.py
```

#### Deep Learning Model (LSTM-GRU-Attention)
```bash
python deep_learning_model_training/model_training_lstm_gru_v2.py
```
Models are saved as `.h5` files with corresponding logs per fold.

---

## Data

The dataset includes 700 samples (350 truthful, 350 deceitful) collected from 35 participants. Each JSON file captures cursor movements, timestamps, and derived behavioral features:

- **Trajectory:** `mouseMovements`, `timestamps`, `velocity`, `acceleration`, `jerk`, `curvature`
- **Behavioral:** `pausePoints`, `hesitation`, `hesitationLevel`, `totalTime`, `averageSpeed`
- **Metadata:** `question`, `answer`, `label` (`0` = truthful, `1` = deceitful)

### Data Flow

<p align="center">
  <img src="extra_img/data_flow_diagram.png" alt="Data Flow" width="90%" />
</p>

The diagram shows the flow of mouse data from collection to JSON storage.

---

## Methodology

Participants answered yes/no questions truthfully and deceitfully in a browser-based experiment. Mouse dynamics were recorded in real time using JavaScript and saved as JSON via a Node.js backend.

Raw (x, y) cursor paths were processed into standardized feature sets including movement derivatives and behavioral summaries. Models were trained using stratified 5-fold cross-validation.<br />

### Classical Model Architecture

<p align="center">
  <img src="extra_img/classical_model_architecture.png" alt="Data Flow" width="90%" />
</p>

The diagram illustrates data processing and evaluation flow for the Random Forest model.<br />

### Deep Learning Architecture

<p align="center">
  <img src="extra_img/dl_model_architecture.png" alt="Data Flow" width="90%" />
</p>

The diagram presents the hybrid model combining LSTM, GRU, soft attention, and meta-feature fusion.<br />

### Evaluation
Evaluation included accuracy, recall, macro F1-score, ROC/AUC, matrix correlations and feature importance.<br />

---

## Project Workflow

<p align="center">
  <img src="extra_img/project_flow.png" alt="Data Flow" width="90%" />
</p>

This diagram illustrates the end-to-end workflow of the project, starting from data collection and recording, through preprocessing and feature extraction, branching into both classical (Random Forest) and deep learning (LSTM-GRU-Attention) training pipelines, and culminating in a final comparative evaluation of model performance.

The diagram summarizes data collection, feature extraction, model training, and evaluation.

---

## Results

| Model           | Accuracy | Macro F1 | AUC   |
|----------------|----------|----------|-------|
| Random Forest  | 58.6%    | 0.58     | 0.60  |
| Deep Learning  | 62.1%    | 0.62     | 0.65  |

- Deep learning outperformed classical model across most metrics
- Jerk spikes, hesitation, and pause duration were among the top contributing features
- ROC and PR curves, as well as training visualizations, are available in `graph_charts/` and `classical_graph/`

---

## Contributing

This project was part of a bachelor thesis and is not currently accepting contributions. However, feel free to explore, fork, or build upon it for educational or research purposes.

---

## License

This repository is open for academic and non-commercial use. For reuse or citation, please include a reference to the thesis author and NTNU.

---

## GitHub Stats

![Nemanja's GitHub stats](https://github-readme-stats.vercel.app/api?username=nemanjant&show_icons=true&theme=default&hide=prs)

---

## Thesis

This GitHub repository supports the bachelor thesis submitted to NTNU Gjøvik.  
Repository URL:  
[https://github.com/nemanjant/PROG2900-Bachelor-project-2025](https://github.com/nemanjant/PROG2900-Bachelor-project-2025)



