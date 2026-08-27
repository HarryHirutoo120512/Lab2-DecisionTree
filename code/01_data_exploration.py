"""
VAN DE 1: Chon & Mo ta Dataset
Dataset: Titanic - Machine Learning from Disaster
Nguon: Kaggle / open mirror (datasciencedojo/datasets)
Bai toan: Binary Classification - du doan hanh khach song sot (Survived: 0/1)
"""
import pandas as pd

df = pd.read_csv("../data/titanic.csv")

print("=" * 60)
print("1. TONG QUAN DATASET")
print("=" * 60)
print(f"So mau (rows): {df.shape[0]}")
print(f"So feature (cols, chua tinh target): {df.shape[1] - 1}")
print(f"Cac cot: {list(df.columns)}")

print("\n" + "=" * 60)
print("2. KIEU DU LIEU TUNG COT")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("3. GIA TRI THIEU (MISSING VALUES)")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"missing_count": missing, "missing_%": missing_pct}))

print("\n" + "=" * 60)
print("4. PHAN BO TARGET (Survived)")
print("=" * 60)
print(df["Survived"].value_counts())
print(df["Survived"].value_counts(normalize=True).round(3) * 100, "%")
print(">> Nhan xet: target hoi lech (khong can bang 50/50) -> phu hop de")
print("   thu nghiem ky thuat xu ly class imbalance o phan cai thien model.")

print("\n" + "=" * 60)
print("5. THONG KE MO TA (numeric features)")
print("=" * 60)
print(df.describe())

# Luu ban tom tat ra file text de dua vao report
with open("../outputs/01_dataset_summary.txt", "w") as f:
    f.write("DATASET: Titanic\n")
    f.write(f"So mau: {df.shape[0]}\n")
    f.write(f"So feature goc: {df.shape[1]-1}\n")
    f.write(f"Target: Survived (0 = khong song sot, 1 = song sot)\n")
    f.write(f"Ty le target: {dict(df['Survived'].value_counts(normalize=True).round(3))}\n")
    f.write(f"\nMissing values:\n{missing[missing>0].to_string()}\n")

print("\nDa luu tom tat vao outputs/01_dataset_summary.txt")
