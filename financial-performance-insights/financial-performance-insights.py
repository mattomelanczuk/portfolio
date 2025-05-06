import pandas as pd
import os
from itertools import combinations

# Set the working directory to the location of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Load the dataset
dataset = "deutsche_bank_financial_performance.csv"

df = pd.read_csv(dataset)

print(df.head())
print(df.columns)
print(df.dtypes)

print(df.describe())

# Print the earliest and latest date in the DataFrame
date_column = None
for col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        date_column = col
        break
    # Try to parse if not already datetime
    try:
        pd.to_datetime(df[col])
        date_column = col
        break
    except Exception:
        continue

if date_column:
    df[date_column] = pd.to_datetime(df[date_column])
    print(f"Earliest date: {df[date_column].min()}")
    print(f"Latest date: {df[date_column].max()}")

# Verify 'Debt_to_Equity' is equal to Liabilities / Equity for each entry
if 'Debt_to_Equity' in df.columns and 'Liabilities' in df.columns and 'Equity' in df.columns:
    calculated_ratio = df['Liabilities'] / df['Equity']
    mismatches = df[~df['Debt_to_Equity'].round(2).eq(calculated_ratio.round(2))]
    if mismatches.empty:
        print("All 'Debt_to_Equity' entries are correct.")
    else:
        print("Mismatches found in 'Debt_to_Equity' calculation:")
        print(mismatches[['Liabilities', 'Equity', 'Debt_to_Equity']], calculated_ratio[mismatches.index])
else:
    print("Required columns ('Debt_to_Equity', 'Liabilities', 'Equity') not found in DataFrame.")