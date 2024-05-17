import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 读取CSV文件
df = pd.read_csv('average_velocities_per_cars.csv')

# 初始化上一行的数据
previous_row_data = df.iloc[0]

# 初始化用于存储每一行平均值的列表
average_values = []

# 循环遍历每一行（从第二行开始）
for i in range(1, len(df)):
    # 获取当前行的数据
    current_row_data = df.iloc[i]

    # 计算当前行的平均值
    average_value = current_row_data.mean()
    average_values.append(average_value)
    previous_row_data = current_row_data

# 绘制平均值折线图
plt.plot(list(range(1000, 2, -1)), average_values)
# plt.semilogx(range(2, len(df) + 1), average_values)
plt.ylabel("Average Velocity (m/s)")
plt.xlabel("Interval (ms)(log)")
plt.title("Avg_Spd_Curve")
plt.grid(False)
plt.show()
