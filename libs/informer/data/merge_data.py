import numpy as np
import pandas as pd

# 读取第一张表并解析时间列
df1 = pd.read_csv('table1.csv', encoding='GBK', parse_dates=[0], index_col=[0])

# 读取第二张表并解析时间列（第二列）
df2 = pd.read_csv('table2.csv', encoding='GBK', parse_dates=[1], index_col=[1])
df2 = df2.iloc[:, [df2.columns.get_loc('ICBC')]]

# 读取第三张表并解析时间列（第三列）
df3 = pd.read_csv('table3.csv', encoding='GBK', parse_dates=[2], index_col=[2])
df3 = df3[['换手率(%)', '市盈率', '市净率', '市销率', '市现率']]

# 按索引合并三张表
merged_df = df1.join(df2, how='inner').join(df3, how='inner')

# 重命名索引为 "date"
merged_df.index.name = 'date'
# 重命名字段
merged_df.rename(columns={
    '换手率(%)': 'TurnoverRate',
    '市盈率': 'PERatio',
    '市净率': 'PBRatio',
    '市销率': 'PSRatio',
    '市现率': 'PCRatio'
}, inplace=True)
# 计算 "delteCoVaR" 字段的分位数
quantiles = merged_df['delteCoVaR'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values

# 确保最大值被包括在内，可以通过将最大边界设为无穷大来实现
quantiles[-1] = np.inf

# 使用 pd.cut 来根据分位数创建新的 "risk" 字段
merged_df['risk'] = pd.cut(merged_df['delteCoVaR'], bins=quantiles, labels=[0, 1, 2, 3], include_lowest=True)
# 保存结果到新的 CSV 文件
merged_df.to_csv('merged_table.csv')
