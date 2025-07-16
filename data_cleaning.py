import pandas as pd

df = pd.read_csv('amazon.csv')

# Clean price columns
df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)

# Clean rating count (fill NaN with 0 before converting to int)
df['rating_count'] = df['rating_count'].replace('[,]', '', regex=True).fillna(0).astype(int)

# Check for nulls
print(df.isnull().sum())

# Preview cleaned data
print(df.head())
