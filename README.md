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
├── classical_model_training/         # Random Forest model scripts
├── deep_learning_model_training/     # LSTM-GRU-Attention scripts and logs
├── data/                             # Raw cursor JSON files
├── data_analysys_stats/              # Data processing & visualization scripts
├── public/                           # Frontend for mouse-data collection
├── server.js                         # Node.js backend for JSON storage
├── package.json / package-lock.json  # Node.js dependencies
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

</details>

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/nemanjant/PROG2900-Bachelor-project-2025.git
   cd PROG2900-Bachelor-project-2025
   ```

2. **Install dependencies**

   Ensure Python 3.11 and Node.js are installed:
   ```bash
   npm install
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Collection

Run the local web experiment to record mouse movements:
```bash
node server.js
```
Files are saved under `data/` and should subsequently be organized into `data/truthful/` and `data/deceitful/`.

### 2. Data Analysis & Feature Preparation

Process raw JSON and generate feature visualizations:
```bash
# Summary statistics
python data_analysys_stats/utils/average_mouse_stats.py

# Cursor path interpolation
python data_analysys_stats/utils/average_mouse_pattern_graphs.py

# Behavioral feature plots
python data_analysys_stats/utils/average_mouse_stats_chart.py
# ... other plotting scripts
```

### 3. Model Training

**Classical (Random Forest)**
- Baseline (80/20 split):
  ```bash
  python classical_model_training/model_training_rf_v1.py
  ```
- 5-fold cross-validation:
  ```bash
  python classical_model_training/model_training_rf_cv_fold.py
  ```

**Deep Learning (LSTM-GRU-Attention)**
```bash
python deep_learning_model_training/model_training_lstm_gru_v2.py
```
Models (HDF5) and logs are saved per fold.

---

## Methodology

Raw cursor trajectories (x, y, timestamp) are preprocessed into kinematic features (velocity, acceleration, jerk, curvature) and behavioral summaries (pause points, hesitation).
Stratified cross-validation ensures balanced class splits.

<p align="center">
  <img src="extra_img/data_flow_diagram.png" width="90%" alt="Figure 1: Data flow from collection to JSON storage"/>
  <br/>
  **Figure 1.** Data flow from collection to JSON storage.
</p>

---

## Classical Model Architecture

<p align="center">
  <img src="extra_img/classical_model_architecture.png" width="90%" alt="Figure 2: Random Forest processing and evaluation flow"/>
  <br/>
  **Figure 2.** Random Forest processing and evaluation flow.
</p>

---

## Deep Learning Architecture

<p align="center">
  <img src="extra_img/dl_model_architecture.png" width="90%" alt="Figure 3: LSTM-GRU-Attention hybrid network"/>
  <br/>
  **Figure 3.** LSTM-GRU-Attention hybrid network.
</p>

---

## Project Workflow

<p align="center">
  <img src="extra_img/project_flow.png" width="90%" alt="Figure 4: End-to-end workflow with branching for classical and deep pipelines"/>
  <br/>
  **Figure 4.** End-to-end workflow with branching into classical and deep learning pipelines, concluding in comparative evaluation.
</p>

---

## Results & Evaluation

| Model           | Accuracy | Macro F1 | AUC   |
|-----------------|----------|----------|-------|
| Random Forest   | 58.6%    | 0.58     | 0.60  |
| Deep Learning   | 62.1%    | 0.62     | 0.65  |

- Deep learning significantly outperforms the classical approach.  
- Key predictors include jerk spikes, hesitation duration, and pause patterns.  

### Evaluation Analysis

Confusion matrices and ROC curves reveal that the Random Forest model under-predicts deceit (higher false-negative rate), whereas the LSTM-GRU-Attention network achieves a true positive rate of 0.68 and ROC AUC of 0.65. A McNemar’s test indicates the performance difference is statistically significant (p < 0.05).

---

## Conclusions

This study demonstrates that cursor dynamics contain detectable signals of deceptive intent, with deep sequence models outperforming classical feature-based classifiers. Future work could explore real-time deployment and multi-modal integration.

---

## References

- Breiman, L. (2001). *Random forests*. Machine Learning, 45(1), 5–32.  
- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural Computation, 9(8), 1735–1780.  
- Cho, K. et al. (2014). *On the properties of neural machine translation: Encoder-decoder approaches*. arXiv:1409.1259.  
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural machine translation by jointly learning to align and translate*. arXiv:1409.0473.

---

## Contributing

This work was produced as a bachelor thesis and is not currently seeking contributions. You are welcome to fork and adapt the code for research or educational purposes.

---

## License

Academic and non-commercial use only. Please cite the author and NTNU when reusing.

---

## GitHub Stats

![Nemanja's GitHub stats](https://github-readme-stats.vercel.app/api?username=nemanjant&show_icons=true&theme=default&hide=prs)

---

## Thesis

This repository supports the bachelor thesis submitted to NTNU Gjøvik.  
[https://github.com/nemanjant/PROG2900-Bachelor-project-2025](https://github.com/nemanjant/PROG2900-Bachelor-project-2025)
