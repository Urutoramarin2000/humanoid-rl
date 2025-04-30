import json
import matplotlib.pyplot as plt

# 步骤1: 加载JSON文件
with open('/home/zhengde.ma/humanoid-gym-main/humanoid/scripts/motor_velocity_data.json', 'r') as file:
    double_array = json.load(file)  # 这应该是一个双层数组，包含12个子数组

# 步骤2和3: 提取数据并绘制每条曲线
# 假设双层数组的外层数组的每个元素都是一个包含数据点的列表
x_values = range(len(double_array[0]))  # 通常x值是数据点的索引

plt.figure(figsize=(10, 5))  # 可以指定图形的大小

for i, data in enumerate(double_array):
    plt.plot(x_values, data, label=f'Data {i+1}')  # 绘制每条曲线

# 步骤4: 自定义图表
plt.title('12 Data Sets Plot')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend(title='Datasets')

# 显示图表
plt.show()