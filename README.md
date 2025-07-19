# Heart Disease Prediction using Logistic Regression

This project implements a heart disease classification model using logistic regression from scratch, comparing both non-regularized and L2-regularized approaches. The objective is to predict whether a patient has heart disease based on clinical attributes.

---

## Overview

- **Model**: Logistic Regression (with and without L2 Regularization)
- **Approach**: Custom implementation using NumPy
- **Evaluation**: Accuracy, Cost Analysis, Decision Boundary Visualization
- **Dataset**: UCI Heart Disease dataset

---

## Dataset

The dataset used is the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). It consists of 303 patient records with the following attributes:

**Key Features Used (after preprocessing):**
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- Slope of ST segment
- Number of major vessels
- Thalassemia

Target variable:  
- `1`: Presence of heart disease  
- `0`: Absence of heart disease

---

## Project Structure
```
heart_disease_logistic/
│
├── main.py # Main script to run both models
├── model_utils.py # Functions for cost, gradient, prediction
├── regularize_utils.py # Functions for L2-regularized cost and gradient
├── visual_utils.py # Plotting and decision boundary visualizations
├── heart.csv # Input dataset
├── README.md # Project documentation
└── requirements.txt # Python dependencies
```

---

## Results

| Model                      | Training Accuracy | Test Accuracy | Final Cost |
|---------------------------|-------------------|----------------|------------|
| Logistic Regression       | 87.80%            | 80.98%         | 0.32       |
| Regularized (λ = 0.01)    | 84.51%            | 80.00%         | 0.35       |

**Observations:**
- Regularization slightly reduced overfitting.
- The unregularized model achieved slightly higher training accuracy, but the regularized model had more stable generalization on unseen data.

---

## How to Run

### 1. Clone the Repository
```
git clone https://github.com/MansoobeZahra/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```
2. Install Dependencies

pip install -r requirements.txt
3. Run the Model
```
python main.py
```
This will output training logs, final parameters, accuracy scores, and decision boundary plots.

Dependencies
Python 3.7+

numpy

pandas

matplotlib

scikit-learn

You can install all required packages using:
```
pip install -r requirements.txt
```
License
This project is licensed under the MIT License.
---

## Connect

 By [Mansoob E Zehra](https://github.com/MansoobeZahra)


---
