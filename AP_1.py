import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 1 data collection
df_west = pd.read_csv(r'dataset/haidian-wanliu-air-quality.csv') # It is on the west of Beiijng.
df_east = pd.read_csv(r'dataset/beijing-us embassy-air-quality.csv') # It is on the east of Beijing.

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
df_merged = pd.merge(df_west_sorted, df_east_sorted, on='date', how='outer', suffixes=('_W','_E'))

# 2.4 Test if the dataframe has '' or ' '.
has_empty_strings = (df_merged == '').any().any()
has_spaces = (df_merged == ' ').any().any()
print("Contains empty strings:", has_empty_strings)
print("Contains spaces:", has_spaces)

# 2.5 Identify the NaN values.
df_na = df_merged[(df_merged.isna()).any(axis=1)]
# print(df_na)
# Since there are no records for pm25, pm10, and no2 from the East station in Beijing before May 2018,
# we have decided to begin our analysis from June 2018.
df_merged = df_merged[df_merged['date'] >= '2018-06-01']

# 2.6 outliers
# value_columns = ['pm25_W', 'pm10_W', 'o3_W', 'no2_W', 'pm25_E', 'pm10_E', 'o3_E', 'no2_E']
df_melted = df_merged.melt(id_vars='date', var_name='value_type', value_name='value')
plt.Figure(figsize=(12,8))
sns.boxplot(x='value_type', y='value', data=df_melted)
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# It shows that pm10_W and pm10_E have outliers, which we will check.

top_pm10_W =df_merged.nlargest(3, 'pm10_W')
top_pm10_E =df_merged.nlargest(3, 'pm10_E')
print('\nOutlier Records: \n', pd.concat([top_pm10_W,top_pm10_E]).drop_duplicates())
# The highest three records occurred on March 14, 2021, December 11, 2022, and April 12, 2023.
# We verified these dates online and confirmed that they were significant dates when Beijing experienced strong dust storms
# We confirmed that these 'outliers' are indeed valid data points and will include them in our analysis.

# 2.7 Decide the strategy of data sampling
grouped = df_merged.groupby(pd.Grouper(key='date', freq='M'))
empty_counts = grouped.agg(lambda x: x.isna().sum()) # count the NaN in each column
df_group = empty_counts.reset_index()
print('\nNaN Values in Each Month: \n', df_group)
# Since most months do not have missing values except for '2020-02' and '2021-10', we choose to fill NaN using a linear model.

# 2.8 Complete the data gap using automated linear model.
df_merged.set_index('date', inplace=True)
# Fill in the missing dates between '2018-06-01' and '2024-09-30' if they exist.
df_merged_full = df_merged.reindex(pd.date_range(start=df_merged.index.min(), end=df_merged.index.max(), freq='D'))
df = df_merged_full.interpolate(method='linear') # Fill in the missing values using 'linear'
print('\nThe Cleaned Air Quality Dataset: \n', df.head(10))

# 3 Summary Statistics
print('\nSummary Statistics: \n', df.describe().round(2))

# 4 Time Series Plot
# 4.1 Time Series Plot by day
plt.figure(figsize=(12,6))
sns.lineplot(data=df[['pm25_W','pm10_W','o3_W','no2_W']], marker='.')
plt.xlabel('Date')
plt.ylabel('Indicator')
plt.title('Daily Time Series of West Station (Beijing)')
plt.show()

plt.figure(figsize=(12,6))
sns.lineplot(data=df[['pm25_E','pm10_E','o3_E','no2_E']], marker='.')
plt.xlabel('Date')
plt.ylabel('Indicator')
plt.title('Daily Time Series of Eest Station (Beijing)')
plt.show()

# 4.2 Time Series Plot by month
df_monthly_avg = df.resample('M').mean()

plt.figure(figsize=(12,6))
sns.lineplot(data=df_monthly_avg[['pm25_W','pm10_W','o3_W','no2_W']], marker='o')
plt.xlabel('Date')
plt.ylabel('Indicator')
plt.title('Monthly Time Series of West Station (Beijing)')
plt.show()

plt.figure(figsize=(12,6))
sns.lineplot(data=df_monthly_avg[['pm25_E','pm10_E','o3_E','no2_E']], marker='o')
plt.xlabel('Date')
plt.ylabel('Indicator')
plt.title('Monthly Time Series of East Station (Beijing)')
plt.show()

# 5 Histograms
Column_W = ['pm25_W', 'pm10_W', 'o3_W', 'no2_W']
Column_E = ['pm25_E', 'pm10_E', 'o3_E', 'no2_E']

plt.figure(figsize=(12,10))
for i,col in enumerate(Column_W, 1):
    plt.subplot(2,2,i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
# plt.tight_layout()
plt.suptitle('Distribution of West Station (Beijing)')
plt.show()

plt.figure(figsize=(12,10))
for i,col in enumerate(Column_E, 1):
    plt.subplot(2,2,i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
# plt.tight_layout()
plt.suptitle('Distribution of East Station (Beijing)')
plt.show()

# 6 Correlation Analysis
# 6.1 corr() matrix
cor_matrix_W = df[Column_W].corr()
print('\nCorrelation Coefficients of West Station:\n', cor_matrix_W)

cor_matrix_E = df[Column_E].corr()
print('\nCorrelation Coefficients of East Station:\n', cor_matrix_E)

# 6.2 p value matrix
def cal_p_value(df,columns):
    p_values = pd.DataFrame(index=columns, columns=columns)
    for col1 in columns:
        for col2 in columns:
            _,p_value = stats.pearsonr(df[col1], df[col2])
            p_values.loc[col1, col2] = p_value
    return p_values

pv_matrix_W = cal_p_value(df, Column_W)
pv_matrix_E = cal_p_value(df, Column_E)
print('\np values of West Station:\n', pv_matrix_W)
print('\np values of East Station:\n', pv_matrix_E)

# PM2.5 and PM10 show a correlation with coefficients of 0.36 (West Station) and 0.4 (East Station), both with p-values less than 0.05.
# NO2 and PM10 are correlated with coefficients of 0.34 (West Station) and 0.39 (East Station), both with p-values less than 0.05.

# 7 Trend Analysis
# 7.1 Time Series of pm2.5
plt.figure(figsize=(12,6))
sns.lineplot(x=df.index, y=df['pm25_W'], marker='o', label='PM2.5 (West)')
plt.title('PM2.5 Trend at West Station (Beijing)')
plt.xlabel('Date')
plt.ylabel('PM2.5')
plt.show()

# 7.2 Trend of pm2.5
sns.regplot(x=df.index.factorize()[0], y=df['pm25_W'], scatter=False, label='Trend Line', color='r')
plt.show()

# As the trend line shows, pm2.5 at West Station shows a significant increasing trend.



