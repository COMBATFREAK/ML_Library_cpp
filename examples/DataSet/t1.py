import pandas as pd

# Read CSV file into a DataFrame
df = pd.read_csv('HeartDisease.csv')

# Drop rows with 'NA' values
df_cleaned = df.dropna()

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('cleaned_file.csv', index=False)

# Print the cleaned DataFrame
print(df_cleaned)
