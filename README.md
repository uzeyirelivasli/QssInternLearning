# ❤️ Heart Disease Prediction — Neural Network from Scratch

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only-013243?style=flat&logo=numpy&logoColor=white)
![No Frameworks](https://img.shields.io/badge/No%20TensorFlow%2FPyTorch-built%20from%20scratch-red?style=flat)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat)

A multi-layer perceptron (MLP) implemented **entirely from scratch with NumPy** — no PyTorch, TensorFlow, or Keras — trained to predict the presence of heart disease using the Cleveland Heart Disease dataset.

---

## 📊 Dataset

The [Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) contains 303 patient records with 13 physiological attributes.

| Attribute | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (1–4) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment (0–2) |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia (0–3) |
| `target` | Diagnosis of heart disease (1 = present, 0 = absent) |

---

## 🧠 Model architecture

```
Input (n features)
      │
      ▼
┌─────────────┐
│  Hidden 1   │  6 units · ReLU
└─────────────┘
      │
      ▼
┌─────────────┐
│  Hidden 2   │  4 units · ReLU
└─────────────┘
      │
      ▼
┌─────────────┐
│   Output    │  1 unit · Sigmoid
└─────────────┘
```

| Layer | Weight shape | Bias shape |
|---|---|---|
| Input → Hidden 1 | `(n_x, 6)` | `(1, 6)` |
| Hidden 1 → Hidden 2 | `(6, 4)` | `(1, 4)` |
| Hidden 2 → Output | `(4, 1)` | `(1, 1)` |

Every matrix operation, activation function, forward pass, backward pass (backpropagation via the chain rule), and parameter update was written manually with NumPy — no autodiff, no pre-built layers.

---

## ⚙️ Pipeline

1. **Preprocessing**
   - One-hot encoding for nominal categorical features: `cp`, `restecg`, `slope`, `thal`
   - Train / validation / test split (stratified)
   - `StandardScaler` fit **only on the training set**, then applied to validation and test sets (no data leakage)
2. **Training**
   - Mini-batch gradient descent (shuffled every epoch)
   - Binary cross-entropy loss
   - He initialization for ReLU layers
   - **Early stopping** based on validation loss, to avoid overfitting
3. **Evaluation**
   - Accuracy, confusion matrix on the held-out test set

---

## 📈 Results

```
Early stopping at epoch 44 (best val_cost = 0.3675)

Test accuracy: 86.8%

Confusion matrix:
              Predicted: 0    Predicted: 1
Actual: 0         85              15
Actual: 1         12              93
```

|  | Value |
|---|---|
| Accuracy | 86.8% |
| Precision | 86.1% |
| Recall | 88.6% |
| F1-score | 87.3% |

---

## ❓ Theory questions

**1. Matrix dimensions and mini-batch training**

For a mini-batch of size `m`:

```
X   : (m, n_x)
W1  : (n_x, 6)     b1 : (1, 6)     →  A1 : (m, 6)
W2  : (6, 4)       b2 : (1, 4)     →  A2 : (m, 4)
W3  : (4, 1)       b3 : (1, 1)     →  A3 : (m, 1)
```

Instead of passing the entire training set through the network at once (batch gradient descent) or one row at a time (stochastic gradient descent), the training set is shuffled every epoch and split into fixed-size mini-batches. Each batch runs a full forward/backward pass, and the parameters are updated after every batch — giving faster, less noisy, and more memory-efficient convergence.

**2. When to stop training**

The validation loss is tracked after every epoch. Training continues as long as it keeps improving. If it fails to improve for a set number of consecutive epochs (`patience`), training stops early and the weights from the best-performing epoch are restored. This prevents overfitting — training loss can keep dropping while validation loss rises, which is the sign the model has started memorizing the training data instead of generalizing.

---

## 🚀 How to run

```bash
pip install numpy pandas scikit-learn
python Heart_diseases.py
```

Make sure `heart_disease_dataset.csv` (or `heart.csv`) is in the same directory as the script.

---

## 📁 Repository structure

```
heart_diseases_2_hidden/
├── Heart_diseases.py     # main script — data pipeline + NN from scratch
├── heart_disease_dataset.csv
└── README.md
```

---

*Built as a hands-on practice project — Cleveland Heart Disease dataset, MLP from scratch, no deep learning frameworks.*
