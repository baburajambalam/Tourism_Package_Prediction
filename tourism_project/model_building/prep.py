import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Data Cleaning: normalize categorical inconsistencies
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
if "Designation" in df.columns:
    df["Designation"] = df["Designation"].replace({"Senior Manager": "Sr. Manager"})

# Drop non-predictive identifiers and unnecessary legacy index columns
drop_cols = [c for c in ["CustomerID", "Unnamed: 0"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits in both current working directory and model_building directory
save_targets = [".", "tourism_project/model_building"]
for st in save_targets:
    if os.path.exists(st):
        Xtrain.to_csv(os.path.join(st, "Xtrain.csv"), index=False)
        Xtest.to_csv(os.path.join(st, "Xtest.csv"), index=False)
        ytrain.to_csv(os.path.join(st, "ytrain.csv"), index=False)
        ytest.to_csv(os.path.join(st, "ytest.csv"), index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
