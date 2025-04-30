import json
import matplotlib.pyplot as plt
import numpy as np

# 读取 JSON 数据
with open('/home/zhengde.ma/humanoid-gym-main/humanoid/scripts/data/actions_data.json', 'r') as file:
    double_array = json.load(file)  # 这应该是一个双层数组，包含12个子数组

# 转置数据，以便按列绘制
data_transposed = list(zip(*double_array))

# 定义一个函数来计算数值导数
def numerical_derivative(data, dx=1.0):
    der = [(data[i + 1] - data[i]) / dx for i in range(len(data) - 1)]
    return der

# 计算每个序列的二阶导数
second_derivatives = []
for column in data_transposed:
    # 将数据转换为 numpy 数组
    data = np.array(column)
    # 计算一阶导数
    first_derivative = numerical_derivative(data)
    # 计算二阶导数
    second_derivative = numerical_derivative(first_derivative)
    second_derivatives.append(second_derivative)

# 计算子图的行数和列数
num_columns = len(second_derivatives)
num_rows = (num_columns + 1) // 2  # 确保有足够的行来容纳所有的列

# 创建一个图表，并为每一列绘制一条曲线
plt.figure(figsize=(15, num_rows * 4))  # 调整图表大小以容纳所有子图
for i, derivative in enumerate(second_derivatives):
    plt.subplot(num_rows, 2, i+1)  # 创建子图
    plt.plot(derivative, label=f'Second Derivative {i+1}')
    plt.legend()
    plt.title(f'Second Derivative {i+1} - Time Series Data')

# 调整子图间距
plt.tight_layout()
# 显示图表
plt.show()
