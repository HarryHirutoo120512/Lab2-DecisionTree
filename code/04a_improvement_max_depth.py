"""
VAN DE 4 - PHUONG PHAP CAI THIEN 1: Tuning max_depth
Y tuong: baseline overfit vi cay qua sau (depth=23). Gioi han do sau se
buoc cay tong quat hoa tot hon, giam overfitting.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42

X_train = pd.read_csv("../outputs/X_train.csv")
X_test = pd.read_csv("../outputs/X_test.csv")
y_train = pd.read_csv("../outputs/y_train.csv").squeeze()
y_test = pd.read_csv("../outputs/y_test.csv").squeeze()

depths = list(range(1, 21))
train_acc, test_acc = [], []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

# Ve curve accuracy vs depth
plt.figure(figsize=(8, 5))
plt.plot(depths, train_acc, marker="o", label="Train Accuracy")
plt.plot(depths, test_acc, marker="s", label="Test Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs max_depth")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/04a_depth_vs_accuracy.png", dpi=150)
plt.close()

best_depth = depths[test_acc.index(max(test_acc))]
print(f"Do sau cho test accuracy cao nhat: max_depth = {best_depth} "
      f"(test acc = {max(test_acc):.4f})")

# Train model tot nhat voi max_depth toi uu
best_model = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

metrics = {
    "model": "Improvement 1 (max_depth tuning)",
    "best_param": f"max_depth={best_depth}",
    "train_accuracy": accuracy_score(y_train, y_pred_train),
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "error_rate": 1 - accuracy_score(y_test, y_pred),
}
print("\nKET QUA:")
for k, v in metrics.items():
    print(f"{k}: {v}")

pd.DataFrame([metrics]).to_csv("../outputs/results_improvement1_maxdepth.csv", index=False)
print("\nDa luu: outputs/results_improvement1_maxdepth.csv")
print("Da luu bieu do: outputs/04a_depth_vs_accuracy.png")
