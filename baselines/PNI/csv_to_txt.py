import pandas as pd
import sys

# Read the CSV file
df = pd.read_csv(sys.argv[1])

first_column = df.iloc[:, 0]

with open(sys.argv[2], 'w') as file:
    for value in df.iloc[:, 0]:
        file.write(f"{value}\n")
