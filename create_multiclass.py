import pandas as pd
import os

folder_path = r"C:\Users\User\Desktop\IDS-Viz\data\CIC-BCCC"

# Use 2000 rows per file = ~20,000 total rows (will be under 100MB)
MAX_ROWS_PER_FILE = 2000

files = {
    'Benign Traffic.csv': 'BENIGN',
    'DDoS ICMP Flood.csv': 'DDOS_ICMP',
    'DDoS UDP Flood.csv': 'DDOS_UDP',
    'DoS ICMP Flood.csv': 'DOS_ICMP',
    'DoS TCP Flood.csv': 'DOS_TCP',
    'DoS UDP Flood.csv': 'DOS_UDP',
    'MITM ARP Spoofing.csv': 'MITM_ARP',
    'Recon Port Scan.csv': 'RECON_PORT',
    'Recon Ping Sweep.csv': 'RECON_PING',
    'Recon Vulnerability Scan.csv': 'RECON_VULN'
}

all_data = []

for filename, label in files.items():
    filepath = os.path.join(folder_path, filename)
    if os.path.exists(filepath):
        print(f"✅ Loading: {filename} -> {label}")
        df = pd.read_csv(filepath, nrows=MAX_ROWS_PER_FILE)
        df['Attack_Type'] = label
        all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = r"C:\Users\User\Desktop\IDS-Viz\cic_multiclass_20k.csv"
combined.to_csv(output_path, index=False)

print(f"\n✅ Saved: {len(combined):,} rows")
print(f"   Classes: {combined['Attack_Type'].nunique()}")
print(f"   File: {output_path}")