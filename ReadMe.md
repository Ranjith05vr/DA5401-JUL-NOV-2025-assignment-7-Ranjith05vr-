# DA5401 – Assignment 7: Multi-Class Model Selection (UCI Satimage Dataset)

This repository contains the Colab-ready notebook and code for **DA5401 Assignment 7** on model selection and evaluation using the **UCI Satimage (Landsat Satellite)** dataset.

---

## 📘 Dataset
- **Source:** [UCI Machine Learning Repository – Statlog (Landsat Satellite)](https://archive.ics.uci.edu/ml/datasets/statlog+(landsat+satellite))
- Files:
  - `sat.trn` → training set (4435 samples)
  - `sat.tst` → test set (2000 samples)
- Each sample contains **36 numerical attributes** representing the 3 × 3 pixel neighbourhood spectral values of four spectral bands.
- Target classes (6 land-cover types):
  1. Red soil  
  2. Cotton crop  
  3. Grey soil  
  4. Damp grey soil  
  5. Soil with vegetation stubble  
  7. Very damp grey soil  

---

## ⚙️ Pipeline Overview

1. **Data Loading & Pre-processing**  
   - Training and test files loaded separately (no cross-validation).  
   - Class labels {1, 2, 3, 4, 5, 7} remapped to {0 – 5}.  
   - Standardization performed using `StandardScaler`.  
   - Visualizations: class distribution bar plot and PCA scatter plot.  

2. **Models Trained**
   - K-Nearest Neighbors (KNN)  
   - Decision Tree  
   - Gaussian Naive Bayes  
   - Logistic Regression (OvR)  
   - Support Vector Classifier (SVC)  
   - Dummy Classifier (Prior based)  
   - RandomForest Classifier  
   - XGBoost Classifier  
   - RandomModel (bad → random probabilities)  
   - InvertedRandomModel (bad → flipped probabilities for AUC < 0.5 demo)

3. **Metrics Evaluated**
   - Accuracy  
   - Weighted F1-Score  
   - Macro-Averaged ROC AUC  
   - Macro-Averaged Average Precision (AP)

4. **Visualizations (black-and-white friendly)**
   - Class distribution bar chart  
   - PCA 2D scatter plot  
   - ROC and PR curves with distinct markers (`-o`, `--x`, `-s`, etc.)  
   - Confusion matrices  
   - Feature importance (RandomForest)

---

## 📊 Final Results

| Model | Accuracy | Weighted F1 | Macro AUC | Macro AP |
|:------|-----------:|-------------:|-----------:|-----------:|
| **RandomForest** | **0.9115** | **0.9094** | **0.9899** | **0.9517** |
| **XGBoost** | 0.9050 | 0.9030 | 0.9898 | 0.9509 |
| **KNN** | 0.9045 | 0.9037 | 0.9780 | 0.9217 |
| **SVC** | 0.8955 | 0.8925 | 0.9847 | 0.9175 |
| **Logistic Regression** | 0.8210 | 0.7935 | 0.9537 | 0.8116 |
| **Gaussian NB** | 0.7965 | 0.8036 | 0.9546 | 0.8105 |
| **Decision Tree** | 0.8505 | 0.8509 | 0.9001 | 0.7366 |
| **InvertedRandomModel (bad)** | 0.1630 | 0.1687 | 0.5144 | 0.1693 |
| **Dummy (prior)** | 0.2305 | 0.0864 | 0.5000 | 0.1667 |
| **RandomModel (bad)** | 0.1530 | 0.1581 | 0.5094 | 0.1632 |

---

## 🧠 Interpretation

- **Best performers:**  
  RandomForest and XGBoost achieved the highest performance across all metrics (AUC ≈ 0.99, AP ≈ 0.95), proving excellent generalization and non-linear feature learning.

- **Strong contenders:**  
  KNN and SVC also performed well, indicating that both instance-based and margin-based approaches effectively separated the classes.

- **Moderate models:**  
  Logistic Regression and Gaussian NB were constrained by linear and independence assumptions but still achieved reasonable results.

- **Overfitting sign:**  
  Decision Tree performed decently but with a lower macro-AP (0.73), hinting at overfitting.

- **Baselines:**  
  Dummy and Random models represent chance-level performance (AUC ≈ 0.5).  
  The InvertedRandomModel intentionally misranks class probabilities to produce AUC values near or slightly below 0.5.  
  Both confirm that trained models significantly outperform random guessing.

---

## 🏁 Conclusion

The **RandomForest Classifier** was selected as the final model for its consistently high accuracy (91%), strong macro-AUC (≈ 0.99), and balanced average precision (≈ 0.95).  
All other trained classifiers also outperformed random baselines, confirming successful model selection and evaluation in line with the assignment objectives.

---

## 🧩 Requirements
- python>=3.8
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- xgboost


---

## ▶️ How to Run (Google Colab)

1. Upload `sat.trn` and `sat.tst` files to your Google Colab working directory.  
2. Open the assignment notebook or paste the code from this repository into a new Colab notebook.  
3. Run cells sequentially (A → B → C → D).  
4. Generated outputs include:
   - Baseline model metrics  
   - ROC and PR plots (macro-averaged)  
   - Confusion matrices  
   - Summary tables of all evaluation metrics  
5. Optionally re-run the RandomModel and InvertedRandomModel cells to observe AUC fluctuations around 0.5.

---

## 👨‍💻 Author

**Ranjith.V.R**  
*M.S. Research Scholar – IIT Madras*  
**Course:** DA5401 – Machine Learning  
**Assignment:** A7 – Model Selection using UCI Satimage Dataset  

---

