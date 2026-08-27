"""
VAN DE 2: Baseline Decision Tree Model
- Tien xu ly du lieu
- Chia train/test
- Train baseline model (tham so mac dinh)
- Visualize cay
- Danh gia: Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import joblib

RANDOM_STATE = 42

# ---------- 1. LOAD & PREPROCESS ----------
df = pd.read_csv("../data/titanic.csv")

# Chon feature: bo cac cot dinh danh khong co gia tri du doan (ID, Name, Ticket, Cabin qua nhieu missing)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

df = df[features + [target]].copy()

# Xu ly missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical -> numeric (Decision Tree sklearn can numeric)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

X = df[features]
y = df[target]

print("Sau tien xu ly - kiem tra missing:")
print(X.isnull().sum())

# ---------- 2. TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------- 3. TRAIN BASELINE MODEL (tham so mac dinh) ----------
baseline = DecisionTreeClassifier(random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)

# ---------- 4. VISUALIZE CAY ----------
plt.figure(figsize=(22, 12))
plot_tree(
    baseline, feature_names=features, class_names=["Not Survived", "Survived"],
    filled=True, rounded=True, fontsize=7, max_depth=3  # chi ve 3 tang dau cho de nhin
)
plt.title("Baseline Decision Tree (hien thi 3 tang dau)")
plt.tight_layout()
plt.savefig("../outputs/02_baseline_tree.png", dpi=150)
plt.close()
print("Da luu hinh cay: outputs/02_baseline_tree.png")

# ---------- 5. DANH GIA ----------
y_pred_train = baseline.predict(X_train)
y_pred_test = baseline.predict(X_test)
y_proba_test = baseline.predict_proba(X_test)[:, 1]

def evaluate(y_true, y_pred, y_proba=None, label=""):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    result = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    if y_proba is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_proba)
    print(f"\n--- {label} ---")
    for k, v in result.items():
        print(f"{k:10s}: {v:.4f}")
    print(f"{'error_rate':10s}: {1-acc:.4f}")
    return result

print("\n" + "=" * 60)
print("BASELINE MODEL - KET QUA")
print("=" * 60)
train_metrics = evaluate(y_train, y_pred_train, label="TRAIN SET")
test_metrics = evaluate(y_test, y_pred_test, y_proba_test, label="TEST SET")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Survived", "Survived"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Baseline Model (Test set)")
plt.tight_layout()
plt.savefig("../outputs/02_baseline_confusion_matrix.png", dpi=150)
plt.close()
print("\nDa luu confusion matrix: outputs/02_baseline_confusion_matrix.png")

print(f"\nDo sau cay (tree depth): {baseline.get_depth()}")
print(f"So la (leaves): {baseline.get_n_leaves()}")

# Luu model va du lieu split de cac script sau dung lai
joblib.dump(baseline, "../outputs/baseline_model.joblib")
X_train.to_csv("../outputs/X_train.csv", index=False)
X_test.to_csv("../outputs/X_test.csv", index=False)
y_train.to_csv("../outputs/y_train.csv", index=False)
y_test.to_csv("../outputs/y_test.csv", index=False)

# Luu bang ket qua
results_summary = pd.DataFrame([
    {"model": "Baseline", "dataset": "train", **train_metrics},
    {"model": "Baseline", "dataset": "test", **test_metrics},
])
results_summary.to_csv("../outputs/results_baseline.csv", index=False)
print("\nDa luu ket qua: outputs/results_baseline.csv")
