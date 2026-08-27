"""
VAN DE 4 - PHUONG PHAP CAI THIEN 2: Doi splitting criterion (Gini vs Entropy)
+ tang min_samples_leaf de giam overfitting.
Y tuong: (a) so sanh 2 criterion pho bien; (b) ket hop tang min_samples_leaf
de cay khong tao la qua nho (chi 1-2 mau), giup tong quat hoa tot hon.
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42

X_train = pd.read_csv("../outputs/X_train.csv")
X_test = pd.read_csv("../outputs/X_test.csv")
y_train = pd.read_csv("../outputs/y_train.csv").squeeze()
y_test = pd.read_csv("../outputs/y_test.csv").squeeze()

# ---------- (a) So sanh Gini vs Entropy (tham so khac giu mac dinh) ----------
print("=" * 60)
print("(a) SO SANH GINI vs ENTROPY (tham so con lai mac dinh)")
print("=" * 60)
for criterion in ["gini", "entropy"]:
    m = DecisionTreeClassifier(criterion=criterion, random_state=RANDOM_STATE)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"criterion={criterion:8s} -> test accuracy = {acc:.4f}")

# ---------- (b) Grid search: criterion + min_samples_leaf + max_depth ----------
print("\n" + "=" * 60)
print("(b) GRID SEARCH: criterion x min_samples_leaf x max_depth")
print("=" * 60)
param_grid = {
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [1, 5, 10, 20, 30],
    "max_depth": [3, 5, 7, 10, None],
}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

metrics = {
    "model": "Improvement 2 (criterion + min_samples_leaf tuning)",
    "best_param": str(grid.best_params_),
    "train_accuracy": accuracy_score(y_train, y_pred_train),
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "error_rate": 1 - accuracy_score(y_test, y_pred),
}
print("\nKET QUA TREN TEST SET:")
for k, v in metrics.items():
    print(f"{k}: {v}")

pd.DataFrame([metrics]).to_csv("../outputs/results_improvement2_criterion.csv", index=False)
print("\nDa luu: outputs/results_improvement2_criterion.csv")
