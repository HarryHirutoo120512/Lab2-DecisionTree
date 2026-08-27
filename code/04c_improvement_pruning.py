"""
VAN DE 4 - PHUONG PHAP CAI THIEN 3: Cost-Complexity Pruning (ccp_alpha)
Y tuong: dung sklearn's cost_complexity_pruning_path de tim chuoi cac cay
da duoc "cat tia" (pruned) va chon alpha cho test accuracy cao nhat.
Day la ky thuat pruning chinh thong, khac voi 2 phuong phap tren (dieu chinh
tham so truoc khi train), pruning hoat dong SAU khi cay day du da duoc xay dung.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42

X_train = pd.read_csv("../outputs/X_train.csv")
X_test = pd.read_csv("../outputs/X_test.csv")
y_train = pd.read_csv("../outputs/y_train.csv").squeeze()
y_test = pd.read_csv("../outputs/y_test.csv").squeeze()

# ---------- 1. Tinh duong dan cost-complexity pruning ----------
full_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
path = full_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Bo alpha cuoi cung (thuong tuong ung voi cay chi co 1 node)
ccp_alphas = ccp_alphas[:-1] if len(ccp_alphas) > 1 else ccp_alphas
print(f"So luong ccp_alpha candidates: {len(ccp_alphas)}")

# ---------- 2. Dung 5-fold CV tren TRAIN de chon alpha (khong nhin test set) ----------
# Lay mau bot alpha (buoc nhay) de giam so lan train, van du dai dien
sampled_idx = np.linspace(0, len(ccp_alphas) - 1, min(30, len(ccp_alphas))).astype(int)
sampled_alphas = ccp_alphas[sampled_idx]

cv_means, train_scores_full, test_scores_full, models = [], [], [], []
for alpha in ccp_alphas:
    m = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=alpha)
    m.fit(X_train, y_train)
    train_scores_full.append(accuracy_score(y_train, m.predict(X_train)))
    test_scores_full.append(accuracy_score(y_test, m.predict(X_test)))
    models.append(m)

cv_scores = []
for alpha in sampled_alphas:
    m = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=alpha)
    scores = cross_val_score(m, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, train_scores_full, marker="o", label="Train Accuracy", alpha=0.6)
plt.plot(ccp_alphas, test_scores_full, marker="s", label="Test Accuracy", alpha=0.6)
plt.plot(sampled_alphas, cv_scores, marker="^", label="5-fold CV Accuracy (Train)", linewidth=2)
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.title("Accuracy vs ccp_alpha (Cost-Complexity Pruning)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/04c_pruning_alpha_vs_accuracy.png", dpi=150)
plt.close()

# ---------- 3. Chon alpha tot nhat theo CV (tranh nhin thang vao test set) ----------
best_cv_idx = int(np.argmax(cv_scores))
best_alpha = sampled_alphas[best_cv_idx]
best_model = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=best_alpha)
best_model.fit(X_train, y_train)
print(f"Best alpha chon theo 5-fold CV (mean acc = {cv_scores[best_cv_idx]:.4f})")

y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

print(f"\nBest ccp_alpha = {best_alpha:.5f}")
print(f"Do sau sau khi pruning: {best_model.get_depth()} "
      f"(so voi baseline depth=23)")
print(f"So la sau khi pruning: {best_model.get_n_leaves()} "
      f"(so voi baseline 153 la)")

metrics = {
    "model": "Improvement 3 (Cost-Complexity Pruning)",
    "best_param": f"ccp_alpha={best_alpha:.5f}",
    "tree_depth": best_model.get_depth(),
    "n_leaves": best_model.get_n_leaves(),
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

pd.DataFrame([metrics]).to_csv("../outputs/results_improvement3_pruning.csv", index=False)
print("\nDa luu: outputs/results_improvement3_pruning.csv")
print("Da luu bieu do: outputs/04c_pruning_alpha_vs_accuracy.png")
