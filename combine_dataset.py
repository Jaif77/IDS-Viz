import pandas as pd

# Load both files
benign = pd.read_csv('Benign Traffic.csv')
attack = pd.read_csv('DDoS ICMP Flood.csv')

# Add label column
benign['Label'] = 'BENIGN'
attack['Label'] = 'DDoS'

# Combine
combined = pd.concat([benign, attack], ignore_index=True)

# Shuffle
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
combined.to_csv('cic_iomt_combined.csv', index=False)

print(f"✅ Combined dataset: {len(combined)} rows")
print(f"   Benign: {len(benign)}")
print(f"   Attack: {len(attack)}")