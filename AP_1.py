import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 1 data collection
df_west = pd.read_csv(r'dataset/haidian-wanliu-air-quality.csv')
df_east = pd.read_csv(r'dataset/beijing-us embassy-air-quality.csv')

# print(df_west.head())
# print(df_east.head())

# 2 data cleaning

# Adjust the dtype to datetime
df_west['date'] = pd.to_datetime(df_west['date'])
df_east['date'] = pd.to_datetime(df_east['date'])

# Sorted by Datetime
df_west_sorted = df_west.sort_values(by='date', ascending=True, inplace=False)
df_east_sorted = df_east.sort_values(by='date', ascending=True, inplace=False)

# Add the column 'station'
# df_west_sorted['station'] = 'west'
# df_east_sorted['station'] = 'east'

# Merge two tables
df_merged = pd.merge(df_west_sorted, df_east_sorted, on='date', how='inner', suffixes=('_W','_E'))
# print(df_merged.head(100))

# Delete the space of column names and values
df_merged.columns = df_merged.columns.str.strip()
# df_merged = df_merged.map(lambda x: x.strip() if isinstance(x, str) else x)
df_2 = df_merged[df_merged['pm25_E'] == ' ']
print(df_2)

# Drop NA
# df_cleaned = df_merged.dropna(how='any', subset='pm25_E')
# print(df_cleaned.head(10))



