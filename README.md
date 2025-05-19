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

<br/>

This project looks at whether you can spot when someone’s lying by tracking tiny changes in their mouse movements while they answer "Yes" and "No" questions. Two kinds of models were build and tested —a Random Forest using hand-picked features and a combined LSTM-GRU network with attention—to see which one works better.

---

## Table of Contents

<br/>

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

## Motivation

<br/>

Detecting deception in real-time interactions is critical for enhancing security in online banking systems, fraud detection, and secure authentication processes. Traditional identity verification and fraud prevention methods—such as knowledge-based questions, one-time passwords, or physiological sensors—can be bypassed or require extra hardware. Mouse cursor dynamics provide a lightweight, non-intrusive behavioral biometric that captures subtle motor and cognitive signatures. By analyzing features like movement speed, acceleration, curvature, and hesitation, this study investigates whether deceptive intent during simple yes/no prompts can be inferred to bolster banking security and user authentication.

---

## Repository Structure

<br/>

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
├── extra_img                         # GitHub readme diagrams
├── package.json / lock               # Node.js config and dependencies
├── requirements.txt                  # Python dependencies
├── LCENSE                            # License
└── README.md                         # Project documentation (you are here)
```
</details>

---

## Installation

<br/>

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

<br/>

### 1. Data Collection
To run the local web experiment:
```bash
node server.js
```
Mouse movement responses will be saved as structured JSON files. After collection, sort files into `data/truthful/` and `data/deceitful/`.

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

# Cumputing confidence intervals and paired t-test analysis
confidence_interval.py
t_test.py
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

<br/>

The dataset includes 700 samples (350 truthful, 350 deceitful) collected from 35 participants. Each JSON file captures cursor movements, timestamps, and derived behavioral features:

- **Trajectory:** `mouseMovements`, `timestamps`, `velocity`, `acceleration`, `jerk`, `curvature`
- **Behavioral:** `pausePoints`, `hesitation`, `hesitationLevel`, `totalTime`, `averageSpeed`
- **Metadata:** `question`, `answer`, `label` (`0` = truthful, `1` = deceitful)

<br/>

### Data Flow

<br/>

<p align="center">
  <img src="extra_img/data_flow_diagram.png" alt="Data Flow" width="90%" />
</p>

The diagram shows the flow of mouse data from collection to JSON storage.

---

## Methodology

<br/>

Participants answered "Yes" and "No" questions truthfully and deceitfully in a browser-based experiment. Mouse dynamics were recorded in real time using JavaScript and saved as JSON via a Node.js backend. Raw (x, y) cursor paths were processed into standardized feature sets including movement derivatives and behavioral summaries. Models were trained using stratified 5-fold cross-validation.<br />

<br/>

### Classical Model Architecture

<br/>

<p align="center">
  <img src="extra_img/classical_model_architecture.png" alt="Data Flow" width="90%" />
</p>

The diagram illustrates data processing and evaluation flow for the Random Forest model.<br />

<br/>

### Deep Learning Architecture

<br/>

<p align="center">
  <img src="extra_img/dl_model_architecture.png" alt="Data Flow" width="90%" />
</p>

The diagram presents the hybrid model combining LSTM, GRU, soft attention, and meta-feature fusion.<br />

<br/>

### Evaluation

<br/>

To thoroughly assess model performance and interpretability, we used a comprehensive set of evaluation measures:

- **Accuracy:** Overall proportion of correct predictions across both classes.  
- **Recall (Sensitivity):** Class-specific true positive rate for both truthful and deceitful classes.  
- **Precision:** Proportion of positive predictions that are correct, highlighting the trade-off with recall.  
- **Macro F1-Score:** Harmonic mean of precision and recall, averaged equally across classes to account for any imbalance.  
- **ROC Curve & AUC:** Receiver Operating Characteristic curve and its area measure the model’s discrimination ability across classification thresholds.  
- **Precision-Recall Curve & AP:** Illustrates precision vs. recall trade-offs, especially informative for imbalanced datasets.  
- **Confusion Matrix:** Detailed breakdown of true/false positives and negatives to pinpoint specific error types.  
- **Feature Importance:** Ranking of the top predictive features to support interpretability and behavioral insights.  
- **Correlation Matrix:** Analysis of pairwise feature relationships to detect multicollinearity and guide feature selection.

---

## Project Workflow

<br/>

<p align="center">
  <img src="extra_img/project_flow.png" alt="Data Flow" width="90%" />
</p>

This diagram illustrates the end-to-end workflow of the project, starting from data collection and recording, through preprocessing and feature extraction, branching into both classical (Random Forest) and deep learning (LSTM-GRU-Attention) training pipelines, and culminating in a final comparative evaluation of model performance.

---

## Results & Evaluation

<br/>

### Collected Data Insights

- **Trajectory Patterns:** Truthful paths are smoother and more direct, while deceptive responses exhibit greater curvature, longer detours, and spatial deviation.  
- **Movement Duration:** Truthful trials averaged 2.48 seconds compared to 2.07 seconds for deceptive trials, indicating quicker termination of deceptive actions.  
- **Speed:** Average velocity was lower for truthful responses (355.7 px/s) than for deceptive ones (377.7 px/s), suggesting faster cursor motions when participants lied.  
- **Acceleration:** Deceptive trajectories show higher acceleration peaks in the early phase and larger fluctuations throughout, whereas truthful movement accelerations stabilize more quickly.  
- **Curvature:** Initial curvature is similar, but toward the end of the trajectory, deceptive paths curve more sharply, reflecting irregular directional changes.  
- **Jerk & Spikes:** Deceptive trials produced roughly 12 jerk spikes (vs. 6 in truthful), and peak jerk values were higher, indicating more abrupt motion changes.  
- **Hesitation & Pauses:** Truthful responses had more hesitation events (3.23 vs. 2.53) and pauses (5.18 vs. 4.10), despite equal average pause duration (0.62 seconds), suggesting more deliberate processing when truthful.  

<br/>

## Best Fold Comparison

<div align="center">

| Metric        | Random Forest (Baseline) | Deep Model (Fold 4) |
|---------------|--------------------------|----------------------|
| Accuracy      | 0.586                    | **0.621**            |
| Macro F1      | 0.582                    | **0.621**            |
| AUC           | 0.620                    | **0.650**            |

</div>

The deep learning model outperformed the Random Forest in Fold 4 across all metrics (+3.5% accuracy, +0.039 macro F1, +0.03 AUC).

## Fold-wise Macro F1 Comparison

<div align="center">

| Fold | Random Forest | Deep Learning |
|------|----------------|----------------|
| 1    | **0.614**      | 0.578          |
| 2    | **0.579**      | 0.492          |
| 3    | **0.517**      | 0.514          |
| 4    | 0.571          | **0.621**      |
| 5    | **0.599**      | 0.520          |
| **Avg** | **0.576**  | 0.545          |

</div>

Random Forest shows **greater consistency** and higher average macro F1. Deep model peaks higher, but with **larger fold-to-fold variation**.

## Random Forest 5-Fold Results (with 95% Confidence)

<div align="center">

| Metric               | Mean  | 95% CI     |
|----------------------|--------|------------|
| Accuracy             | 0.577 | ±0.044     |
| Macro F1             | 0.576 | ±0.046     |
| Recall (truthful)    | 0.591 | ±0.048     |
| Recall (deceitful)   | 0.563 | ±0.103     |
| AUC                  | 0.620 | ±0.041     |

</div>

The **tight confidence intervals** for accuracy and truthful recall suggest **stable generalization** across folds.  
The **wider interval** for deceitful recall reflects **greater variability**, possibly due to class imbalance or inconsistency in deceptive behavior.  
The relatively narrow **AUC interval (±0.041)** reinforces the model’s **consistent class-separation capability** across validation splits.

## Deep Learning 5-Fold Results (with 95% Confidence)

<div align="center">

| Metric               | Mean  | 95% CI     |
|----------------------|--------|------------|
| Accuracy             | 0.544 | ±0.068     |
| Macro F1             | 0.545 | ±0.066     |
| Recall (truthful)    | 0.544 | ±0.074     |
| Recall (deceitful)   | 0.566 | ±0.128     |
| AUC                  | 0.574 | ±0.067     |

</div>

The **tightest confidence intervals** are seen for accuracy and macro F1, suggesting reasonably consistent generalization across folds.  
However, the **wide interval for deceitful recall (±0.128)** indicates more variation in how the model detects deception across validation splits—possibly due to differences in deceptive patterns between folds.  
The **AUC margin (±0.067)** reflects a **moderate but stable class separation ability** under different training conditions.


## Statistical Testing

**Paired t-test on Macro F1-scores:** p = 0.288  
The difference in performance between the models is **not statistically significant**.  
The **overlapping 95% confidence intervals** support this conclusion, indicating that any observed difference may be due to random variation across folds.


## Confusion Matrix Comparison

**Random Forest:** 28 false positives, 30 false negatives  
**Deep Learning (Fold 4):** 26 false positives, 27 false negatives  
The deep model had slightly **fewer misclassifications** in its best fold.

## ROC Curve Comparison

<div align="center">

| Model              | AUC  |
|--------------------|------|
| Random Forest      | 0.62 |
| Deep Learning (F4) | 0.65 |

</div>

The deep model achieved a slightly higher AUC (**0.65**) than the Random Forest (**0.62**), suggesting improved ability to distinguish between truthful and deceitful responses.

## Summary

- **Deep Model excels in optimal conditions**, capturing complex behavior dynamics with higher peak performance.
- **Random Forest is more stable**, generalizing reliably across folds with smaller fluctuations.
- **No statistically significant advantage** for either model — both have complementary strengths.
- **Practical takeaway:** Use Random Forest when **data is limited** or consistency is needed; deep learning shines when **rich sequential data and tuning** are available.

---

## Conclusions

- This project investigated whether mouse movement dynamics can be used to detect deception in binary yes/no tasks.

- Two types of models were developed and compared:
  - A **Random Forest classifier** using handcrafted features like speed, acceleration, and curvature.
  - A **deep learning model** based on **LSTM–GRU with attention**, designed to capture temporal dependencies in mouse movement behavior.

- Key findings:
  - The **deep learning model** achieved the highest fold-level performance (accuracy: 62.1%, macro F1: 0.621).
  - The **Random Forest model** had slightly higher average performance across all 5 folds and showed better consistency under limited data.
  - A **paired t-test** found **no statistically significant difference** between the models, highlighting the importance of average performance over best-case results.

- Evaluation approach:
  - **5-fold stratified cross-validation** to ensure balanced and fair evaluation.
  - Computation of **95% confidence intervals** to assess metric stability.
  - **Statistical significance testing** to validate model comparisons.

- Ethical and practical considerations:
  - Deception detection tools must address **false positives**, **user consent**, and **fairness**—especially in **sensitive settings like online banking**.
  - Systems should include safeguards such as **manual review**, **transparent explanations**, and **appeal mechanisms** for flagged users.

- Implications:
  - Mouse dynamics offer a **scalable**, **non-intrusive**, and **cost-effective** way to infer user intent.
  - With **larger datasets**, **advanced hybrid models** (e.g., CNN–RNN), and **richer behavioral features**, performance could be further improved.
  - The study lays the groundwork for future deception detection systems in real-world digital environments.

---

## References

<br/>

- C. Mazza, M. Monaro, F. Burla, M. Colasanti, G. Orrù, S. Ferracuti, and P. Roma, “Use of mouse-tracking software to detect faking-good behavior on personality questionnaires: An explorative study,” *Scientific Reports*, vol. 10, p. 4835, 2020. doi:10.1038/s41598-020-61636-5. [Online]. Available: https://doi.org/10.1038/s41598-020-61636-5

- M. Pusara and C. E. Brodley, “User re-authentication via mouse movements,” in *Proceedings of the 2004 ACM Workshop on Visualization and Data Mining for Computer Security (VizSEC/DMSEC)*, Washington, DC, USA: ACM, 2004, pp. 1–8, isbn:1-58113-974-8. doi:10.1145/1029208.1029210. [Online]. Available: https://www.researchgate.net/publication/221325920

- P. Zimmermann, S. Guttormsen, B. Danuser, and P. Gomez, “Affective computing – a rationale for measuring mood with mouse and keyboard,” *International Journal of Occupational Safety and Ergonomics (JOSE)*, vol. 9, no. 4, pp. 539–551, 2003. doi:10.1080/10803548.2003.11076589. [Online]. Available: https://doi.org/10.1080/10803548.2003.11076589

- S. Khan, C. Devlen, M. Manno, and D. Hou, “Mouse dynamics behavioral biometrics: A survey,” *ACM Computing Surveys*, vol. 37, no. 4, Article 111, pp. 1–32, 2023. doi:10.48550/arXiv.2208.09061. [Online]. Available: https://doi.org/10.1145/3640311

- M. Zuckerman, B. M. DePaulo, and R. Rosenthal, “Verbal and nonverbal communication of deception,” in *Advances in Experimental Social Psychology*, L. Berkowitz, Ed., vol. 14, Academic Press, 1981, pp. 1–59.[Online]. Available: https://doi.org/10.1016/S0065-2601(08)60369-X

- M. Monaro, L. Gamberini, and G. Sartori, “Spotting faked identities via mouse dynamics using complex questions,” in *Proceedings of the British HCI 2018*, Belfast, UK: BCS Learning and Development Ltd., 2018, pp. 1–9. doi:10.14236/ewic/HCI2018.8. [Online]. Available: http://dx.doi.org/10.14236/ewic/HCI2018.8

- N. Siddiqui, R. Dave, M. Vanamala, and N. Seliya, “Machine and deep learning applications to mouse dynamics for continuous user authentication,” *Machine Learning and Knowledge Extraction*, vol. 4, no. 1, pp. 1–24, 2022. doi:https://doi.org/10.3390/make4020023. [Online]. Available: https://doi.org/10.48550/arXiv.2205.13646

- S. Raschka, How to compute confidence intervals for machine learning model metrics, Accessed: 2025-05-18, 2022. [Online]. Available: https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html.

- Stat Trek, Paired sample hypothesis test, Accessed: 2025-05-18, 2025. [Online]. Available: https://stattrek.com/hypothesis-test/paired-means


---

## Contributing

<br/>

This project was part of a bachelor thesis and is not currently accepting contributions. However, feel free to explore, fork, or build upon it for educational or research purposes.

---

## License

<br/>

All rights to the thesis and associated materials are governed by the NTNU Standard Agreement on Student Assignments with External Parties (2020), signed on 24.04.2025.

---

## GitHub Stats

<br/>

![Nemanja's GitHub stats](https://github-readme-stats.vercel.app/api?username=nemanjant&show_icons=true&theme=default&hide=prs)

---

## Thesis

<br/>

This GitHub repository supports the bachelor thesis submitted to NTNU Gjøvik.  
Repository URL:  
[https://github.com/nemanjant/PROG2900-Bachelor-project-2025](https://github.com/nemanjant/PROG2900-Bachelor-project-2025)
