import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('average_speeds_by_multi-density2.csv')

# 列名
density_columns = data.columns

# 存储拟合结果
fit_results = {}

# 遍历每个密度列
for col in density_columns:
    # plt.close()

    # 创建一个新的图形窗口
    plt.figure(figsize=(6, 4))

    # 获取数据
    density_data = data[col]

    # 检查是否有缺失值，并去除
    density_data = density_data[~np.isnan(density_data)]

    # 检查数据是否为空
    if density_data.empty:
        continue

    # 拟合正态分布
    fit_params = stats.norm.fit(density_data)

    # 存储拟合结果
    fit_results[col] = {
        'distribution': 'Normal',
        'parameters': fit_params
    }

    # 绘制直方图
    plt.hist(density_data, bins=20, density=True, alpha=0.7, color='b', label='Histogram')

    # 生成拟合分布的x和y值
    x = np.linspace(min(density_data), max(density_data), 100)
    y = stats.norm.pdf(x, *fit_params)

    # 绘制拟合的分布曲线
    plt.plot(x, y, 'r-', label='Fitted Distribution (Normal)')

    plt.title(f'Fit for {col}')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()

    # 显示当前图形窗口
    plt.show()

for density, result in fit_results.items():
    print(f"Fit for {density}:")
    print(f"Distribution: {result['distribution']}")
    print(f"Parameters: Mean = {result['parameters'][0]}, Std = {result['parameters'][1]}")
