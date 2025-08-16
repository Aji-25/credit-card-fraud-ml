# Credit Card Fraud Detection — PyTorch (Colab)

This project is an end-to-end **binary classification** model that detects fraudulent credit card transactions using a **feed-forward neural network (MLP)** built in PyTorch. The workflow runs in **Google Colab** and includes preprocessing, training, evaluation, and metrics reporting.

---

## ✳️ Project Features

- **Dataset:** Kaggle *Credit Card Fraud Detection* dataset (`creditcard.csv`)
  - Features: anonymized PCA variables `V1..V28`, plus `Time`, `Amount`
  - Label: `Class` (0 = genuine, 1 = fraud)
- **Two training strategies:**
  1. **Balanced subset (undersampling)** — equal fraud and non-fraud samples for fast prototyping.
  2. **Full dataset with class weighting** — train on all data while compensating for imbalance.
- **Model:** Multi-layer perceptron (MLP) in PyTorch.
- **Loss function:** `BCEWithLogitsLoss` (binary cross entropy with logits).
- **Metrics:** Precision, Recall, F1-score, Confusion Matrix.
- **Scaling:** `StandardScaler` applied to `Time` and `Amount`.

---

## 🗂️ Project Files

- `creditfraud.ipynb` — Colab notebook with full code.
- `creditfraud_extracted.py` — optional script with concatenated code cells.

---

## ✅ Requirements

- Python 3.10+ (Colab recommended)
- Libraries:  
  - `torch`, `torchvision`, `torchaudio`  
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`

Install in Colab:

```python
!pip install torch torchvision torchaudio
!pip install scikit-learn matplotlib tqdm pandas numpy
```

---

## 📦 Dataset

Download from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and upload to Colab.

Upload command:

```python
from google.colab import files
uploaded = files.upload()  # select creditcard.csv
```

---

## 🔬 Workflow

1. **Load and Explore Data**
   - Inspect dataset shape, preview rows, check fraud distribution.
2. **Balanced Subset Training**
   - Undersample majority class to match fraud cases.
   - Train/test split, train MLP, evaluate metrics.
3. **Feature Scaling**
   - Apply `StandardScaler` to `Time` and `Amount`.
4. **Full Dataset Training**
   - Train on imbalanced dataset using `BCEWithLogitsLoss(pos_weight=...)`.
   - Evaluate metrics again.

---

## 🧠 Model Details

- **Architecture:**  
  `Linear → ReLU → Linear → ReLU → Linear (1 output logit)`
- **Optimizer:** Adam
- **Loss:** Binary cross-entropy with logits
- **Threshold:** 0.5 for fraud classification
- **Metrics:** Precision, Recall, F1-score, Confusion Matrix

---

## 📊 Metrics Reported

- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-score:** Harmonic mean of Precision & Recall
- **Confusion Matrix:**

```
[[ TN, FP ],
 [ FN, TP ]]
```

---

## ▶️ How to Run

1. Open the notebook in Google Colab.
2. Install required libraries.
3. Upload `creditcard.csv`.
4. Run cells in order:
   - Load + EDA
   - Balanced subset training
   - Scale features
   - Full dataset training

---

## 📌 Results

### Balanced Subset (Undersampling)
- Precision: `XX.XX`
- Recall: `XX.XX`
- F1-score: `XX.XX`
- Confusion Matrix:
  ```
  [[ TN, FP ],
   [ FN, TP ]]
  ```

### Full Dataset (Class Weighting)
- Precision: `XX.XX`
- Recall: `XX.XX`
- F1-score: `XX.XX`
- Confusion Matrix:
  ```
  [[ TN, FP ],
   [ FN, TP ]]
  ```

---

## 🙌 Acknowledgements

- Dataset: *Credit Card Fraud Detection* by MLG-ULB (Kaggle).  
- Libraries: PyTorch, scikit-learn, pandas, numpy, matplotlib.  
