import pandas as pd

# 假设 CSV 文件的路径为 “NSL/combined_data_with_class.csv”
csv_path = r"NSL/combined_data_with_class.csv"

# 读取 CSV 文件
df = pd.read_csv(csv_path)

# 统计 'subclass' 列里每个类型出现的次数
subclass_counts = df['subclass'].value_counts()

print(subclass_counts)
