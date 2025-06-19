import pandas as pd

# Define input and output paths
input_path = 'data/XAU_5m_data.csv'         # Full large file
output_path = 'data/XAU_5m_data_bar.csv'  # Output trimmed file

# Load only needed columns, and last 200,000 rows
print("🔄 Reading CSV...")
df = pd.read_csv(input_path, sep=';', engine='python')
print(f"✅ Loaded {len(df):,} rows.")

# Trim to last 200,000 rows
trimmed_df = df.tail(900_000)

# Save trimmed data
trimmed_df.to_csv(output_path, sep=';', index=False)
print(f"✅ Saved trimmed data to: {output_path}")
print(f"📦 Rows in trimmed file: {len(trimmed_df):,}")
