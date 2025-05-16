import pandas as pd
import sys

# Load the two CSV files
csv1 = pd.read_csv(sys.argv[1])
csv2 = pd.read_csv(sys.argv[2])

merged_csv = pd.merge(csv1, csv2, on='Path', how='outer', suffixes=('_csv1', ''))

for col in csv1.columns:
    if col != 'Path' and col in csv2.columns:
        merged_csv.drop(f"{col}_csv1", axis=1, inplace=True)

merged_csv.to_csv(sys.argv[3], index=False)

print(f"CSV files merged successfully into '{sys.argv[3]}'")
