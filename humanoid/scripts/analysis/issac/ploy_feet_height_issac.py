import json
import matplotlib.pyplot as plt

# 假设JSON文件名为'data.json'
with open('/home/zhengde.ma/humanoid-gym-main/humanoid/scripts/data/feet_height_data.json', 'r') as file:
    # 读取JSON文件并解析为Python列表
    data = json.load(file)
# 查看  angle_velocity_data
# 查看feet高度： feet_height_data
# 查看feet smoothness_data: feet_height_smooth_list
# 查看air time： air_time_data
# feet_dis_list
# 将JSON中的数值转换为Python列表
velocity_values = data

# 绘制曲线图
plt.figure(figsize=(10, 5))  # 设置图形大小
plt.plot(velocity_values, label='Feet height')  # 绘制曲线
plt.title('Feet height over Time')  # 设置图形标题
plt.xlabel('Time')  # 设置x轴标签
plt.ylabel('Feet height')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 显示图形
