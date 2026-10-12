# Customer Churn Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting customer churn using Machine Learning techniques. Customer churn occurs when customers stop using a company's services. Predicting churn can help businesses identify customers who may be at risk of leaving.

The project uses the Telco Customer Churn dataset to explore customer information, train a classification model, and evaluate its performance.

## 🎯 Objectives

- Analyze customer data to understand churn patterns.
- Prepare the dataset for Machine Learning.
- Train a classification model to predict customer churn.
- Tune model hyperparameters using GridSearchCV.
- Compare model performance before and after tuning.

## 📊 Dataset

**Dataset:** Telco Customer Churn

**Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains customer information, subscribed services, account details, billing information, and the churn status.

The target variable is `Churn`, which indicates whether a customer has left the service.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (if used in the notebook)
- Google Colab
- Machine Learning classification techniques

## ⚙️ Project Workflow

1. **Data Loading:** Load the Telco Customer Churn dataset.
2. **Data Preprocessing:** Inspect missing values, duplicates, data types, and data quality.
3. **Exploratory Analysis:** Examine customer attributes and churn distribution, where applicable.
4. **Feature Preparation:** Prepare the features and target variable for model training.
5. **Model Training:** Train the initial classification model.
6. **Hyperparameter Tuning:** Use GridSearchCV to search for better model settings.
7. **Evaluation:** Compare model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
8. **Comparison:** Review the original and tuned model results.

## 📈 Results

The notebook reports the following accuracy values:

| Model | Accuracy |
|---|---:|
| Original model | 73.81% |
| Tuned model | 73.81% |

The tuned model did not improve overall accuracy in this experiment. Further work could focus on feature engineering, class imbalance, and selecting evaluation metrics suited to churn prediction.

## 🚀 How to Run

1. Open the notebook in Google Colab.
2. Download the dataset from the Kaggle source linked above.
3. Upload the CSV file to your Colab environment.
4. Update the dataset path if necessary.
5. Run the notebook cells in order.

## 🔮 Future Improvements

- Improve recall for customers who churn.
- Explore class imbalance handling.
- Try alternative classification algorithms.
- Perform additional feature engineering.
- Compare models using ROC-AUC and precision-recall metrics.
- Improve generalization through systematic validation.

## 👨‍💻 Author

**Tanush Kumar**

B.Tech — Computer Science and Engineering (AI & ML)

Graduation Year: 2028
