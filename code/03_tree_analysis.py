"""
VAN DE 3: Phan tich cay quyet dinh (Baseline)
- Feature importance
- Nhan xet cau truc cay (do sau, so la)
- Danh gia overfit/underfit
"""
import pandas as pd
import matplotlib.pyplot as plt
import joblib

baseline = joblib.load("../outputs/baseline_model.joblib")
X_train = pd.read_csv("../outputs/X_train.csv")

features = list(X_train.columns)

# ---------- 1. FEATURE IMPORTANCE ----------
importances = pd.Series(baseline.feature_importances_, index=features).sort_values(ascending=False)
print("=" * 60)
print("FEATURE IMPORTANCE (Baseline Tree)")
print("=" * 60)
print(importances)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="steelblue")
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Feature Importance - Baseline Decision Tree")
plt.tight_layout()
plt.savefig("../outputs/03_feature_importance.png", dpi=150)
plt.close()
print("\nDa luu: outputs/03_feature_importance.png")

# ---------- 2. NHAN XET CAU TRUC CAY ----------
depth = baseline.get_depth()
n_leaves = baseline.get_n_leaves()
n_nodes = baseline.tree_.node_count

print("\n" + "=" * 60)
print("CAU TRUC CAY")
print("=" * 60)
print(f"Do sau (depth): {depth}")
print(f"So node: {n_nodes}")
print(f"So la (leaves): {n_leaves}")
print(f"Feature quan trong nhat (root split): {importances.index[0]} "
      f"(importance = {importances.iloc[0]:.3f})")

# ---------- 3. GHI NHAN XET (dua vao report) ----------
notes = f"""
PHAN TICH CAY BASELINE
========================
1. Feature quan trong nhat: {importances.index[0]} ({importances.iloc[0]:.3f}),
   theo sau la {importances.index[1]} ({importances.iloc[1]:.3f}).
   -> Phu hop voi kien thuc thuc te: gioi tinh (Sex) va gia ve (Fare)/hang ghe (Pclass)
      la yeu to anh huong lon nhat den kha nang song sot tren Titanic
      ("Women and children first").

2. Do sau cay = {depth}, so la = {n_leaves} tren chi {len(X_train)} mau train.
   -> Cay QUA SAU / QUA PHUC TAP so voi kich thuoc du lieu.
   -> Nhieu la chi chua 1-2 mau (cay "hoc thuoc long" tung truong hop training).

3. So sanh Accuracy Train (98.17%) vs Test (82.12%):
   -> Chenh lech ~16% -> day la dau hieu OVERFITTING ro ret.
   -> Nguyen nhan: khong gioi han max_depth, min_samples_leaf mac dinh = 1
      nen cay tiep tuc split cho den khi moi la thuan khiet (pure).

=> KET LUAN: can ap dung cac ky thuat regularization (gioi han do sau,
   tang min_samples_leaf, pruning...) o phan cai thien model de giam overfitting
   va tang kha nang tong quat hoa (generalization) tren du lieu moi.
"""
print(notes)

with open("../outputs/03_tree_analysis_notes.txt", "w") as f:
    f.write(notes)
print("Da luu ghi chu phan tich: outputs/03_tree_analysis_notes.txt")
