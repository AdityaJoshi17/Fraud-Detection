# 💳 Credit Card Fraud Detection Project

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
