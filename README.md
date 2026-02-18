### 🚢 Titanic Survival Prediction: End-to-End ML Pipeline
This repository contains a comprehensive machine learning workflow for the Kaggle Titanic competition. The project focuses on robust data cleaning, feature engineering, and comparing ensemble models to predict passenger survival.

## 📌 Project Highlights
⦁	Strategic Imputation: Used grouped statistics to fill missing ages, preserving demographic trends.
⦁	Leakage Prevention: Split data before encoding and used Mutual Information for feature selection.
⦁	Outlier Management: Applied clipping to the Fare feature to stabilize model training.
⦁	Feature Engineering: Created a FamilySize feature to capture social dynamics on board.

## 🛠️ Data Preprocessing & Cleaning
1. Handling Missing Values
The initial dataset showed significant missingness: Age (20%), Cabin (77%), and Embarked (2 passengers).

⦁	Dropped Features: PassengerId, Name, Ticket, and Cabin. (Cabin was removed due to the 77% missing rate).
⦁	Embarked: Imputed using the most frequent value (Mode).
⦁	Age: Imputed by calculating the mean age grouped by Sex and Pclass. This ensures a 1st-class female's age is estimated differently than a 3rd-class male's.

2. Outlier Detection & Treatment
⦁	Age: Visualized via histograms; the distribution appeared natural with no extreme outliers.
⦁	Fare: Analysis via boxplots revealed 116 outliers (13% of data) using the IQR method.
⦁	Solution: Used Clipping to cap extreme fares, preventing them from skewing the model weights.

3. Feature Engineering
⦁	FamilySize: Created by summing SibSp (siblings/spouses) + Parch (parents/children) + 1 (the passenger). This simplified three variables into one meaningful social metric.

## 🧪 Model Selection & Training
To prevent Data Leakage, categorical encoding and feature selection were performed after the train-test split. We used Mutual Information (MI) to identify the most predictive features.


## Model Performance Evaluation

| Model | Train Accuracy | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Decision Tree | 83.99% | 80.45% | 0.83 | 0.62 | 0.71 |
| **Random Forest** | **87.22%** | **83.24%** | **0.81** | **0.74** | **0.77** |
| Gradient Boosting | 89.75% | 81.56% | 0.85 | 0.64 | 0.73 |


## 📊 Key Findings
⦁	Best Model: The Random Forest emerged as the top performer. It achieved the highest Test Accuracy (83.2%) and the most balanced F1-Score (0.77) for survivors.
⦁	Recall vs. Precision: While Gradient Boosting had high precision (0.85), its lower recall (0.64) means it missed more actual survivors compared to the Random Forest.
⦁	Overfitting Check: Gradient Boosting showed the highest training accuracy (89.7%) but a significant drop on the test set, suggesting it was slightly overfitting the training noise.

## 📂 Project Structure

```bash
├── data/
│   ├── raw/
│   │   └── titanic_dataset.csv
│   └── cleaned/
│       └── titanic_dataset_cleaned.csv
├── notebook/
│   └── titanic_survival.ipynb
└── README.md
```
