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
Department of Computer Science, Gjøvik — Spring 2025

---

## GitHub Stats

![Nemanja's GitHub stats](https://github-readme-stats.vercel.app/api?username=nemanjant&show_icons=true&theme=default&hide=prs)

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This repository contains all code and data for the bachelor thesis *"Cursor Dynamics for Deception Detection"*. The study explores whether deceptive intent can be detected from subtle variations in mouse movement behavior during binary question answering. Two modeling approaches were developed: a classical Random Forest model based on handcrafted features, and a deep learning model combining LSTM, GRU, and attention mechanisms.

---

## Repository Structure

```
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

### Data Collection (optional)
To run the local web experiment:
```bash
node server.js
```
Mouse movement responses will be saved to `data/` as structured JSON files. After collection, sort files into `data/truthful/` and `data/deceitful/`.

### Data Analysis
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

### Model Training

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

See `data/sample_schema.md` for full parameter details.

---

## Methodology

Participants answered yes/no questions truthfully and deceitfully in a browser-based experiment. Mouse dynamics were recorded in real time using JavaScript and saved as JSON via a Node.js backend.

Raw (x, y) cursor paths were processed into standardized feature sets including movement derivatives and behavioral summaries. Models were trained using stratified 5-fold cross-validation.

- **Classical Model:** Random Forest on handcrafted features
- **Deep Learning Model:** LSTM + GRU + Attention, using both sequences and meta-features

Evaluation included accuracy, recall, macro F1-score, ROC/AUC, and feature importance.

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

## Thesis

This GitHub repository supports the bachelor thesis submitted to NTNU Gjøvik.  
Repository URL:  
[https://github.com/nemanjant/PROG2900-Bachelor-project-2025](https://github.com/nemanjant/PROG2900-Bachelor-project-2025)



