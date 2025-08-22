# Payment_Anomaly_Detector

# Fraud Detection using Imbalanced Dataset: A Comprehensive Approach

This notebook implements a machine learning pipeline for detecting fraudulent transactions from the **Credit Card Fraud Detection** dataset. The dataset is highly imbalanced, with fraudulent transactions representing a very small fraction of the total transactions(0.17%). To counter this issue, a combination of resampling techniques (SMOTE and undersampling) and weight-based algorithms (LightGBM) are used. The notebook explores the impact of these methods on model performance, evaluates various metrics, and ultimately fine-tunes the model to maximize performance.

## Problem Overview

The **Credit Card Fraud Detection** dataset is characterized by a **highly imbalanced class distribution**, where fraudulent transactions are much less frequent than legitim ones. This imbalance makes it challenging to build a reliable model, as traditional machine learning algorithms tend to bias the majority class (legitimate transactions), resulting in poor performance for the minority class (fraudulent transactions).

## Dataset

This project uses a dataset from Kaggle. You can find the dataset : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


## Approach

The problem of fraud detection was approached in the following steps:

1. **Dataset Preprocessing**:
   - The dataset consists of features such as transaction amount, time, and anonymized features. The target variable is binary, indicating whether a transaction is legitimate (0) or fraudulent (1).
   - The data was first **standardized** and then split into training and testing sets.

2. **Handling Class Imbalance**:
   To address the class imbalance, several strategies were tested:

   - **SMOTE (Synthetic Minority Over-sampling Technique)**:
     - **SMOTE** was initially applied to oversample the minority class by generating synthetic samples based on the feature space of the existing fraud instances. This technique increases the representation of the minority class, which helps the model learn the decision boundaries of fraud cases.
   
   - **Undersampling**:
     - **Undersampling** of the majority class (legitimate transactions) was also attempted to balance the dataset. This technique reduces the number of majority class samples to match the minority class, but it can lead to loss of important information from the majority class, which may cause **underfitting** in the model.

   - **Limitations of Resampling**:
     - The combination of **SMOTE** and **undersampling** led to significant **underfitting**, as the model struggled to generalize on new, unseen data due to the loss of meaningful information from the majority class. While this resampling approach helped balance the dataset, it did not result in a robust fraud detection model.

3. **Weight-Based Algorithm: LightGBM**:
   - As a final step, **LightGBM**, a gradient boosting framework optimized for speed and efficiency, was used. LightGBM allows for the incorporation of **class weights**, which dynamically adjusts the importance of the minority class (fraudulent transactions) during model training, without requiring resampling.
   - By setting the `scale_pos_weight` parameter, we gave more weight to fraudulent transactions, allowing the model to focus on detecting the rare fraud cases effectively while avoiding the loss of information from the majority class.

4. **Model Evaluation**:
   The model was evaluated using multiple metrics to assess its ability to detect fraud, considering the severe class imbalance:

   - **Recall**: Measures the ability of the model to identify all fraudulent transactions (True Positive rate).
   - **Precision**: Measures the proportion of correctly predicted fraud cases relative to all predicted fraud cases.
   - **F1 Score**: The harmonic mean of Precision and Recall, providing a single metric to balance both.
   - **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: A comprehensive metric that measures the model's ability to discriminate between the positive and negative classes. A higher AUC indicates better model performance.

   Additionally, the **Precision-Recall Curve** was plotted to analyze the trade-off between Precision and Recall at different thresholds.

5. **Threshold Tuning**:
   By adjusting the decision threshold, the model's **Precision**, **Recall**, and **F1 Score** were optimized. The goal was to find the optimal threshold that maximizes **F1 Score**, balancing the trade-off between false positives and false negatives.

6. **Final Model Performance**:
   After tuning the threshold, the **LightGBM** model achieved the following performance metrics:
   - **Precision**: 0.99
   - **Recall**: 0.68
   - **F1 Score**: 0.81
   - **ROC-AUC**: 0.9849

   These results demonstrate that the model can effectively identify fraudulent transactions, with **precision** close to 1.0, meaning that the vast majority of fraud predictions are correct. The **F1 score** of 0.81 is impressive, given the significant imbalance in the dataset, and the **ROC-AUC** of 0.9849 indicates excellent overall model performance.

## Key Insights

1. **Resampling Limitations**:
   - **SMOTE** and **undersampling** were effective in balancing the dataset but resulted in **underfitting** due to loss of critical information, particularly from the majority class.
   
2. **LightGBM for Imbalanced Data**:
   - **LightGBM** proved to be a more robust solution for this problem. By using **class weights**, it was able to effectively handle the class imbalance without compromising the model's ability to generalize. This technique significantly improved model performance over the resampling-based approaches.

3. **Model Evaluation and Threshold Optimization**:
   - **Threshold tuning** played a crucial role in improving the model's performance. By adjusting the decision threshold, we were able to maximize the **F1 score**, balancing **Precision** and **Recall** to optimize fraud detection.
   
4. **High-Performance Metrics**:
   - Despite the severe imbalance, the final model achieved a **ROC-AUC of 0.9849**, indicating exceptional discrimination power between fraud and legitimate transactions. Additionally, the **Precision** of 0.99 and **F1 Score** of 0.81 highlight the model's strong ability to identify fraudulent transactions with minimal false positives.

## Conclusion

This notebook demonstrates an effective methodology for fraud detection on highly imbalanced datasets. By combining **SMOTE**, **undersampling**, and **LightGBM with class weighting**, we were able to significantly improve model performance. The final model shows that even in the face of severe class imbalance, it is possible to achieve high **Precision** and **F1 Score**, while maintaining a strong **ROC-AUC**.

## Files

- **cc_EDA.ipynb**: The main notebook that contains all the steps for data preprocessing, model training, evaluation, and optimization.

## Requirements

To run this notebook, you will need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `lightgbm`
- `imbalanced-learn`



