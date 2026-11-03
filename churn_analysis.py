
#-----------------------------------
# Import Libraries
#-----------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

#-----------------------
#understand dataset
#-----------------------
BASE_DIR = Path(__file__).resolve().parent

# Load the dataset
file_path = BASE_DIR / "Data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

#-----------------------------------------
#Basic Chicks
#-----------------------------------------
print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print(df.describe(include="all"))
print(df.describe(include="O"))

print("\nDuplicate Check:")
print(df.duplicated().sum())

#--------------------------
# Data Cleaning
#--------------------------
df.drop_duplicates().copy()

# Convert TotalCharges to numeric
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')

# Remove rows without a target value
df = df.dropna(subset=["Churn"]).copy()

# Save cleaned dataset
df.to_csv("cleaned_customer_churn.csv", index=False)

print("\nCleaned dataset shape:", df.shape)
print("Dataset saved successfully!")

#-----------------
# EDA 
#------------------
sns.set_theme(style="whitegrid")
# Churn distribution
plt.figure(figsize=(7,5))
sns.countplot(data=df,x='Churn')
plt.title("Customer-churn-Distribution")
plt.tight_layout()
plt.savefig("Customer-churn-Distribution.png")
plt.show()

#Churn Percentage
churn_rate=(df['Churn']=="Yes").mean()*100
print(f"\nCustomer Churn rate :{churn_rate: .2f}%")

# Churn by contract type
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("churn_by_contract.png")
plt.show()

# Monthly charges by churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.tight_layout()
plt.savefig("monthly_charges_by_churn.png")
plt.show()

# Tenure by churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Tenure vs Churn")
plt.tight_layout()
plt.savefig("tenure_by_churn.png")
plt.show()

#------------------------------
# Prepare Machine Learning Data
#----------------------------

# CustomerID is an identifier, not a useful predictive feature
X = df.drop(columns=["Churn", "customerID"], errors="ignore")
y = df["Churn"].map({"No": 0, "Yes": 1})

# Separate numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", __import__(
        "sklearn.preprocessing",
        fromlist=["OneHotEncoder"]
    ).OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Split before fitting preprocessing to avoid data leakage
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
# -----------------------------------
# Train Machine Learning Model
# -----------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

print("\n--- MODEL TRAINED ---")

# -----------------------------------
#  Evaluate Model
# -----------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
))

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -----------------------------------
# Identify High-Risk Customers
# -----------------------------------

risk_results = X_test.copy()
risk_results["ActualChurn"] = y_test
risk_results["PredictedChurn"] = y_pred
risk_results["ChurnProbability"] = y_prob

risk_results["RiskLevel"] = pd.cut(
    risk_results["ChurnProbability"],
    bins=[-0.01, 0.30, 0.60, 1.00],
    labels=["Low", "Medium", "High"]
)

risk_results = risk_results.sort_values(
    "ChurnProbability", ascending=False
)

risk_results.to_csv("customer_churn_risk_predictions.csv",
    index=False
)

print("\n--- TOP 10 HIGH-RISK CUSTOMERS ---")
print(risk_results.head(10).to_string(index=False))

df.to_csv("cleaned_customer_churn.csv", index=False)

print("\nCleaned dataset shape:", df.shape)
print("Dataset saved successfully!")

print("\n--- PROJECT COMPLETED ---")
print("Results saved in")

from google.colab import files

files.download("cleaned_customer_churn.csv")