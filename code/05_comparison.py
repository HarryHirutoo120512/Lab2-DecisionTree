"""
VAN DE 5: So sanh & Thao luan ket qua
Tong hop baseline + 3 phuong phap cai thien, ve bieu do so sanh,
xac dinh phuong phap tot nhat.
"""
import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("../outputs/results_baseline.csv")
baseline["error_rate"] = 1 - baseline["accuracy"]
baseline_test = baseline[baseline["dataset"] == "test"].iloc[0]

imp1 = pd.read_csv("../outputs/results_improvement1_maxdepth.csv").iloc[0]
imp2 = pd.read_csv("../outputs/results_improvement2_criterion.csv").iloc[0]
imp3 = pd.read_csv("../outputs/results_improvement3_pruning.csv").iloc[0]

rows = [
    {"Model": "Baseline (default params)", "Param": "default",
     "Train Acc": baseline[baseline["dataset"]=="train"].iloc[0]["accuracy"],
     "Test Acc": baseline_test["accuracy"], "Precision": baseline_test["precision"],
     "Recall": baseline_test["recall"], "F1": baseline_test["f1"],
     "Error Rate": baseline_test["error_rate"]},
    {"Model": "Improvement 1: max_depth tuning", "Param": imp1["best_param"],
     "Train Acc": imp1["train_accuracy"], "Test Acc": imp1["accuracy"],
     "Precision": imp1["precision"], "Recall": imp1["recall"], "F1": imp1["f1"],
     "Error Rate": imp1["error_rate"]},
    {"Model": "Improvement 2: criterion + min_samples_leaf", "Param": imp2["best_param"],
     "Train Acc": imp2["train_accuracy"], "Test Acc": imp2["accuracy"],
     "Precision": imp2["precision"], "Recall": imp2["recall"], "F1": imp2["f1"],
     "Error Rate": imp2["error_rate"]},
    {"Model": "Improvement 3: Cost-Complexity Pruning", "Param": imp3["best_param"],
     "Train Acc": imp3["train_accuracy"], "Test Acc": imp3["accuracy"],
     "Precision": imp3["precision"], "Recall": imp3["recall"], "F1": imp3["f1"],
     "Error Rate": imp3["error_rate"]},
]

comparison = pd.DataFrame(rows)
comparison_display = comparison.copy()
for col in ["Train Acc", "Test Acc", "Precision", "Recall", "F1", "Error Rate"]:
    comparison_display[col] = comparison_display[col].round(4)

print("=" * 100)
print("BANG SO SANH TONG HOP")
print("=" * 100)
print(comparison_display.to_string(index=False))

comparison.to_csv("../outputs/05_comparison_table.csv", index=False)

# ---------- Bieu do so sanh Test Accuracy ----------
plt.figure(figsize=(10, 6))
colors = ["gray", "steelblue", "orange", "green"]
bars = plt.bar(comparison["Model"], comparison["Test Acc"], color=colors)
plt.ylabel("Test Accuracy")
plt.title("So sanh Test Accuracy giua Baseline va cac phuong phap cai thien")
plt.xticks(rotation=20, ha="right")
plt.ylim(0.6, 0.9)
for bar, val in zip(bars, comparison["Test Acc"]):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
              ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/05_comparison_accuracy.png", dpi=150)
plt.close()

# ---------- Bieu do so sanh Train vs Test (kiem tra overfit) ----------
plt.figure(figsize=(10, 6))
x = range(len(comparison))
width = 0.35
plt.bar([i - width/2 for i in x], comparison["Train Acc"], width, label="Train Acc", color="salmon")
plt.bar([i + width/2 for i in x], comparison["Test Acc"], width, label="Test Acc", color="seagreen")
plt.xticks(list(x), comparison["Model"], rotation=20, ha="right")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy - Kiem tra muc do Overfitting")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/05_comparison_overfit_check.png", dpi=150)
plt.close()

# ---------- Xac dinh best model ----------
best_row = comparison.loc[comparison["Test Acc"].idxmax()]
gap = comparison["Train Acc"] - comparison["Test Acc"]
comparison["Overfit Gap"] = gap.round(4)

print("\n" + "=" * 60)
print(f"PHUONG PHAP TOT NHAT (theo Test Accuracy): {best_row['Model']}")
print(f"Test Accuracy = {best_row['Test Acc']:.4f}")
print("=" * 60)

discussion = f"""
THAO LUAN KET QUA
==================
1. Baseline (khong gioi han) dat Test Accuracy = {baseline_test['accuracy']:.4f} nhung
   Train Accuracy = {baseline[baseline['dataset']=='train'].iloc[0]['accuracy']:.4f}
   -> khoang cach train-test rat lon -> OVERFIT nang.

2. Improvement 1 (max_depth={imp1['best_param']}) cai thien Test Accuracy len
   {imp1['accuracy']:.4f}, dong thoi giam khoang cach train-test dang ke
   -> cho thay gioi han do sau giup mo hinh tong quat hoa tot hon.

3. Improvement 2 (criterion+min_samples_leaf, chon qua GridSearchCV) cho
   Test Accuracy = {imp2['accuracy']:.4f}. Du CV score tren train cao, nhung
   ket qua tren test khong vuot troi baseline -> minh hoa ro rang: chon
   tham so toi uu theo CV khong luon dam bao tot nhat tren tap test thuc te
   (do bien thien ngau nhien / do phuc tap cua grid).

4. Improvement 3 (Cost-Complexity Pruning, alpha chon qua 5-fold CV) giam
   manh do phuc tap cay (tu depth=23, 153 la -> depth=10, 24 la) trong khi
   Test Accuracy ({imp3['accuracy']:.4f}) van o muc chap nhan duoc va cay
   don gian, de giai thich hon nhieu.

=> KET LUAN: '{best_row["Model"]}' cho Test Accuracy cao nhat trong so cac
   phuong phap thu nghiem. Tuy nhien, neu uu tien tinh don gian/kha nang
   giai thich (interpretability) cua cay, Cost-Complexity Pruning la lua chon
   can bang tot giua accuracy va do phuc tap mo hinh.
"""
print(discussion)

with open("../outputs/05_discussion.txt", "w") as f:
    f.write(discussion)

comparison.to_csv("../outputs/05_comparison_table_full.csv", index=False)
print("\nDa luu: outputs/05_comparison_table.csv, 05_comparison_accuracy.png,")
print("        outputs/05_comparison_overfit_check.png, outputs/05_discussion.txt")
