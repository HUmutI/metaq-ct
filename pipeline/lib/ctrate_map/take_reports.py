import pandas as pd

# Change this to your file path
input_file = "search_CHESTCT.xlsx"
output_file = "just_reports.xlsx"

df = pd.read_excel(input_file)

# Column I is the 9th column (index 8)
col_I = df.iloc[:, 8]

col_I.to_excel(output_file, index=False, header=True)
print(f"Saved {len(col_I)} rows to {output_file}")
