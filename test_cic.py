import pandas as pd
import os

# Find the smallest file to test with (Monday is usually smallest)
csv_files = []
for file in os.listdir("data/MachineLearningCVE/"):
    if file.endswith(".csv"):
        path = f"data/MachineLearningCVE/{file}"
        size = os.path.getsize(path)
        csv_files.append((path, size, file))

# Sort by size and pick smallest
csv_files.sort(key=lambda x: x[1])
smallest_file = csv_files[0][0]
smallest_name = csv_files[0][2]

print(f"📊 Testing with smallest file: {smallest_name}")
print(f"   Size: {csv_files[0][1]/(1024*1024):.1f} MB")

# Load first 10,000 rows
print("🔄 Loading data...")
df = pd.read_csv(smallest_file, nrows=10000)

print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"\n📋 Columns: {df.columns.tolist()[:10]}...")
print(f"\n🏷️ Label distribution:")
print(df[' Label'].value_counts() if ' Label' in df.columns else df.iloc[:,-1].value_counts())