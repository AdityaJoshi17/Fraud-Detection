# 💳 Credit Card Fraud Detection Project

---

Review code : https://colab.research.google.com/drive/1qHO5qHIO23J4-mpmGFf8-m2-G-9TqabQ?usp=sharing

Code With explain : https://colab.research.google.com/drive/1QUucu55H3qtQ_Giqhk591isC4jZZGqKc#scrollTo=EIGiwYsLMoOb

Github Link : https://github.com/AdityaJoshi17/Fraud-Detection/tree/main

---

## 1. ✅ Why is Fraud Detection Important?

Fraud detection is crucial to protect financial systems and customers from unauthorized or suspicious activities. In industries like banking or e-commerce, fraud can lead to significant monetary losses, customer distrust, and legal consequences. Early and accurate detection helps in reducing these risks by flagging suspicious transactions for further investigation.

---

## 2. 📊 Dataset Explanation

The dataset used is the **Credit Card Fraud Detection dataset**. It contains transactions made by credit cards, where:

- Each row is a transaction.
- Most features are anonymized using PCA for confidentiality (`V1` to `V28`).
- Two key features are:
  - `Amount`: The transaction amount.
  - `Class`: The target variable — `0` for normal transactions and `1` for fraudulent transactions.

> **Note**: The dataset is **highly imbalanced**, with very few fraud cases compared to normal ones.

---

## 3. 🌲 What is XGBoost and Why We Used It?

**XGBoost** (Extreme Gradient Boosting) is a fast and powerful gradient boosting algorithm that builds decision trees sequentially, where each new tree corrects the errors of the previous one.

### Why we used it:

- Handles imbalanced data well.
- Efficient and scalable for large datasets.
- Provides good performance with less parameter tuning.
- Supports GPU acceleration for faster training.

In our project, XGBoost helps capture patterns in transaction features to identify fraud effectively.

---

## 4. 🧠 What is DNN? (Deep Neural Network)

**DNN (Deep Neural Network)** is an artificial neural network with multiple layers between the input and output layers, capable of learning complex patterns.

### In our implementation:

- We used a **Sequential** neural network.
- The architecture includes:
  - **Input layer** (shape = number of features)
  - **3 Dense (fully connected) layers** with 64, 32, and 16 units respectively
  - **Dropout layers** after each dense layer to prevent overfitting
  - **Output layer** with 1 neuron and **sigmoid activation** (for binary classification)

> **Total Layers:** 3 hidden dense layers + dropout layers + 1 output layer

---

## 5. 🔀 How Have You Ensembled It?

The two models (XGBoost and DNN) are ensembled using a **soft voting strategy**:

- Both models generate prediction probabilities.
- The probabilities from each model are averaged:

  \[
  \text{Final Prediction} = \frac{\text{XGBoost Prediction} + \text{DNN Prediction}}{2}
  \]

- The final classification is based on this average score.

This improves performance by leveraging the strengths of both models.

---

## 6. 📈 What is AUPRC and Why Have You Used It?

**AUPRC (Area Under the Precision-Recall Curve)** is a performance metric for binary classification, especially useful for **imbalanced datasets**.

### Why we used it:

- Unlike accuracy or AUC-ROC, AUPRC focuses only on the **positive (fraud) class**.
- It evaluates the trade-off between:
  - **Precision**: How many predicted frauds were actual frauds.
  - **Recall**: How many actual frauds were correctly identified.
- In fraud detection, **false negatives are critical**, so recall is very important.
- AUPRC helps measure this more effectively in such scenarios.

---


## 🔍 What is GridSearchCV?

`GridSearchCV` is a method provided by `scikit-learn` that helps in **automatically tuning the best hyperparameters** for a model.

### ✅ Key Features:
- It tries **all combinations** of hyperparameter values you specify.
- For each combination, it performs **cross-validation** (e.g., 3-fold).
- It **evaluates each combination** using a scoring metric (like accuracy, AUC, or average precision).
- Returns the **best combination** based on model performance.

This is particularly useful in models like XGBoost, where performance depends heavily on the choice of hyperparameters.

---

## 🧠 XGBClassifier with GridSearchCV: Explanation

```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

param = {
    'learning_rate': 0.1,
    'verbosity': 2,
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'scale_pos_weight': scale_pos_weight,
    'n_estimators': 300
}

xgb_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

xgbc = XGBClassifier(**param)

xgbc_cv = GridSearchCV(
    estimator = xgbc,
    param_grid = xgb_grid,
    cv = 3,
    scoring = 'average_precision',
    n_jobs = -1,
    verbose = 2
)

xgbc_cv.fit(X_train, y_train)
print('Best parameters: ', xgbc_cv.best_params_)
print('Best score: ', xgbc_cv.best_score_)


## 📌 Explanation

### 🔧 Fixed Parameters (`param`):

- **`learning_rate`**: Controls the learning speed (`0.1` is a common default).
- **`verbosity`**: Level of output verbosity.
- **`objective`**: Binary classification using logistic regression.
- **`tree_method`**: Uses GPU for faster training (`gpu_hist`).
- **`scale_pos_weight`**: Handles class imbalance by giving more weight to the minority class.
- **`n_estimators`**: Number of boosting rounds (trees).

---

### 📊 Hyperparameter Grid (`xgb_grid`):

- **`max_depth`**: Tree depth (controls model complexity).
- **`min_child_weight`**: Minimum sum of instance weight in child nodes (controls overfitting).
- **`gamma`**: Minimum loss reduction required to make a split.
- **`subsample`**: Fraction of training samples used per tree.
- **`colsample_bytree`**: Fraction of features used per tree.

---

### ⚙️ GridSearchCV Setup:

- Tries **all combinations** of values in `xgb_grid`.
- Performs **3-fold cross-validation** (`cv=3`).
- Uses **Average Precision (AUPRC)** as the evaluation metric (ideal for imbalanced datasets).
- Uses **all available CPU cores** (`n_jobs=-1`) for faster computation.
- Shows detailed output logs (`verbose=2`).

---

### ✅ After Training:

- **`best_params_`**: Shows the best combination of hyperparameters found.
- **`best_score_`**: Shows the best AUPRC score obtained during tuning.

