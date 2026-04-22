# Customer-churn-prediction-ML
End-to-end customer churn prediction using XGBoost and SHAP. Exploratory data analysis, preprocessing, modeling and business insights.

---

_**Preliminary Results**_

![](combined.png)

----

| Model                     | ROC AUC | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---------------------------|---------|-------------------|----------------|------------|
| XGBoost                   | 0.8228  | 0.5357            | 0.7219         | 0.6150     |
| SVM                       | 0.8111  | 0.4910            | 0.7790         | 0.6023     |
| Logistic Regression + PCA | 0.8309  | 0.5029            | 0.7861         | 0.6134     |

------

## EDA


>Python Code



```python
# --- Library imports ---
# pandas and numpy handle all our data manipulation
# matplotlib and seaborn are our visualization tools
# shap lets us explain why the model made each prediction
# sklearn provides the split, encoding, metrics, SVM, and Logistic Regression tools
# xgboost is our primary model
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Load the dataset
# This is the Telco Customer Churn dataset — publicly available on Kaggle
# It contains 7,043 customers and 21 variables about their service and behavior
df = pd.read_csv('/content/drive/MyDrive/Eproducts/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# .info() shows column names, data types, and how many non-null values exist per column
# This is our first health check — we want to spot any missing data or wrong types immediately
df.info()
df.head(5)
```

---


Just made a machine learning and data analysis project from scratch.

The goal was simple: predict which customers are likely to cancel a service before they actually do. A real business problem, solved with real data and real code.

Here's what the process actually looks like 👇
You start with raw data, messy, incomplete, full of decisions to make. Which variables matter? Which ones are noise? What do you do with missing values?
Then comes the modeling part. Picking an algorithm is the easy step. Understanding why it works, how to evaluate it honestly, and how to explain its predictions to someone who doesn't speak Python, that's the real challenge.

A few things I learned along the way:
→ Accuracy alone is a terrible metric. It lies to you when your data is imbalanced.

→ A model that performs well on paper can fail completely in production if you're not careful with how you prepare your data.

→ The most valuable skill isn't writing the code ,  it's being able to explain what the model is actually doing and why it matters for the business.

Machine learning is not magic. It's a lot of careful decisions, one after another.

Full project documentation (Customer Churn Prediction, End to End ML)
https://payhip.com/b/AKxdE
