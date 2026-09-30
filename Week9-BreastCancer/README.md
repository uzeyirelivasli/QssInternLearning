<div align="center">

# 🩺 Breast Cancer Classification with a Neural Network

**A beginner-friendly binary classifier built with Keras Dense layers**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📌 Overview

This project trains a small **fully connected (Dense) neural network** to predict whether a breast tumor is **benign** or **malignant**, based on 30 measurements taken from digitized images of cell nuclei.

It is designed as a clear, step-by-step introduction to the neural network workflow:

`Load data → Split → Scale → Build model → Compile → Train → Evaluate → Predict`

## 📊 Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), included with scikit-learn.

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 (radius, texture, perimeter, area, smoothness, ...) |
| Classes | `0` = malignant (212), `1` = benign (357) |
| Task | Binary classification |

## 🧠 Model Architecture

```
Input (30 features)
      │
Dense(16, ReLU)
      │
Dropout(0.3)
      │
Dense(8, ReLU)
      │
Dense(1, Sigmoid)  →  probability of "benign"
```

| Layer | Output shape | Parameters |
|---|---|---|
| Dense (ReLU) | (None, 16) | 496 |
| Dropout (0.3) | (None, 16) | 0 |
| Dense (ReLU) | (None, 8) | 136 |
| Dense (Sigmoid) | (None, 1) | 9 |
| **Total** | | **641** |

**Training setup**

| Setting | Choice | Why |
|---|---|---|
| Optimizer | Adam | Good default, adapts learning rate |
| Loss | Binary cross-entropy | Standard pair for a sigmoid output |
| Metric | Accuracy | Easy to interpret |
| Regularization | Dropout (0.3) + EarlyStopping | Reduce overfitting on a small dataset |

## 🔄 Workflow

1. **Load** the dataset from scikit-learn
2. **Split** into train (80%) and test (20%) with `stratify=y` to keep class ratios
3. **Scale** features with `StandardScaler` (fit on train only, to avoid data leakage)
4. **Build** the model with `keras.Sequential`
5. **Train** with a validation split and `EarlyStopping`
6. **Evaluate** on the untouched test set
7. **Predict** on individual samples

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install tensorflow scikit-learn pandas numpy jupyter
```

### 3. Run the notebook

```bash
jupyter notebook BreastCancer.ipynb
```

Run the cells from top to bottom (**Kernel → Restart & Run All**).

## 📈 Results

Evaluated on a held-out test set of **114 samples** (20% of the data, never seen during training).

**Overall accuracy: 96.5%** (110 of 114 correct)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Malignant | 0.95 | 0.95 | 0.95 | 42 |
| Benign | 0.97 | 0.97 | 0.97 | 72 |

**Confusion matrix**

|  | Predicted malignant | Predicted benign |
|---|---|---|
| **Actual malignant** | 40 ✅ | 2 ❌ |
| **Actual benign** | 2 ❌ | 70 ✅ |

The model correctly identified 40 of 42 malignant tumors (95.2% recall) and missed 2. In a real medical setting, missed malignant cases (false negatives) are the most costly error, which is why recall matters more than accuracy alone.

Exact numbers vary slightly between runs because of random weight initialization and dropout.

## 💡 Key Takeaways

- **Input size** is fixed by the number of features (30); **hidden layer sizes** are hyperparameters you choose.
- **ReLU** in hidden layers adds non-linearity; **Sigmoid** in the output gives a probability for binary classification.
- The scaler is **fit on the training data only** and then applied to the test data.
- In a medical setting, **recall for the malignant class** matters more than raw accuracy, since missing a malignant tumor is the costly mistake.

## 🔭 Possible Improvements

- [ ] Plot training vs. validation loss curves
- [ ] Add confusion matrix and classification report
- [ ] Track precision and recall during training
- [ ] Tune hidden layer sizes and dropout rate
- [ ] Compare with classical models (Logistic Regression, Random Forest)

<div align="center">

If you found this useful, consider giving the repo a ⭐

</div>
