import pandas as pd
import numpy as np

# 创建一个包含 NaN 值的示例 DataFrame
data = {
    'pm25_W': [1.0, np.nan, 3.5, 4.0, 5.0],
    'pm10_W': [4.0, 5.0, 6.0, 7.0, 8.0]
}
df_merged = pd.DataFrame(data)

# 检查 NaN 值
print("NaN rows in pm25_W:")
print(df_merged[pd.isna(df_merged['pm25_W'])])
