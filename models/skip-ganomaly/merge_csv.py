import pandas as pd
import sys

# Load the two CSV files
csv1 = pd.read_csv(sys.argv[1])
csv2 = pd.read_csv(sys.argv[2])

merged_csv = pd.merge(csv1, csv2, on='Path')

merged_csv.to_csv(sys.argv[3], index=False)

print(f"CSV files merged successfully into '{sys.argv[3]}'")
