import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 1 data collection
df_west = pd.read_csv(r'dataset/haidian-wanliu-air-quality.csv')
df_east = pd.read_csv(r'dataset/beijing-us embassy-air-quality.csv')

# 2 data cleaning

# 2.1 Delete the spaces in the column names, as the original data contain spaces, such as ' pm25' and ' o3'.
df_west.columns = df_west.columns.str.strip()
df_east.columns = df_east.columns.str.strip()

# 2.2 Adjust the data type, and delete the 'so2' and 'co' columns, which have recently contained many NaN values.
df_west.drop(columns = ['so2','co'], inplace=True)
for col in df_west:
    if col == 'date':
        df_west[col] = pd.to_datetime(df_west[col], errors='coerce')
    if col != 'date':
        df_west[col] = pd.to_numeric(df_west[col], errors='coerce')
df_west_sorted = df_west.sort_values(by='date', ascending=True, inplace=False)

df_east.drop(columns = ['so2','co'], inplace=True)
for col in df_east:
    if col == 'date':
        df_east[col] = pd.to_datetime(df_east[col], errors='coerce')
    if col != 'date':
        df_east[col] = pd.to_numeric(df_east[col], errors='coerce')
df_east_sorted = df_east.sort_values(by='date', ascending=True, inplace=False)

# 2.3 Merge two tables
df_merged = pd.merge(df_west_sorted, df_east_sorted, on='date', how='inner', suffixes=('_W','_E'))
print(df_merged['pm25_W'].dtype)

# 2.4 Test if the dataframe has '' or ' '.
has_empty_strings = (df_merged == '').any().any()
has_spaces = (df_merged == ' ').any().any()
print("Contains empty strings:", has_empty_strings)
print("Contains spaces:", has_spaces)

# 2.5 Identify the NaN values.
df_na = df_merged[(df_merged.isna()).any(axis=1)]
print(df_na)
# Since there are no records for pm25, pm10, and no2 from the East station in Beijing between April 2017 and May 2018,
# we have decided to begin our analysis from June 2018.
df_merged = df_merged[df_merged['date'] >= '2018-06-01']

# 2.6 Drop NA
# df_dropna = df_merged.dropna(how='any')
# print(df_dropna)

# 2.6 Decide the strategy of data sampling
df_group = df_merged.groupby(df_merged['date'].dt.strftime('%Y-%m')).\
    agg(lambda x: x.notna().sum()).reset_index()
    # size().reset_index(name='sample_count')

print(df_group)

print('\n Print the sample count of each month: \n', df_group)
print('\n Print the median sample count for each month: ', df_group['sample_count'].median())
print('\n Print the average sample count for each month: ', df_group['sample_count'].mean())
# Since the median sample count for all months is 30.0, and the mean is 28.3, we can analyze the data on a daily basis.
# Additionally, we can use an automated linear model to fill in the gaps.

# 2.8 Complete the data gap using automated linear model.
df = df_merged.fillna()


# data sampling
# df = df_cleaned