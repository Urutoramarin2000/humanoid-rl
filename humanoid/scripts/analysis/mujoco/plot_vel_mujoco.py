import json
import matplotlib.pyplot as plt

# 步骤1: 加载JSON文件
with open('/home/zhengde.ma/cowa-humanoid-gym/humanoid/scripts/data/torque_mujoco.json', 'r') as file:
    torque_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组   # motor_velocity_data
# torque_mujoco
# torque
with open('/home/zhengde.ma/cowa-humanoid-gym/humanoid/scripts/data/dof_pos_list_mujoco.json', 'r') as file:
    dof_pos_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组
# dof_pos_list_mujoco
# dof_pos_list
with open('/home/zhengde.ma/cowa-humanoid-gym/humanoid/scripts/data/action_data_mujoco.json', 'r') as file:
    actions_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组
# action_data_mujoco
# actions_data
with open('/home/zhengde.ma/cowa-humanoid-gym/humanoid/scripts/data/ref_pos_list.json', 'r') as file:
    ref_pos_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组

with open('/home/zhengde.ma/cowa-humanoid-gym/humanoid/scripts/data/motor_velocity_data.json', 'r') as file:
    motor_velocity_data = json.load(file)  # 这应该是一个双层数组，包含12个子数组

# 转置数据，以便按列绘制
torque_data_transposed = list(zip(*torque_data))
dof_pos_data_transposed = list(zip(*dof_pos_data))
ref_pos_data_transposed = list(zip(*ref_pos_data))
actions_data_transposed = list(zip(*actions_data))
motor_velocity_data_transposed = list(zip(*motor_velocity_data))

# 计算子图的行数和列数
num_columns_torque = len(torque_data_transposed)
num_rows_torque = (num_columns_torque + 1) // 2  # 确保有足够的行来容纳所有的列

num_columns_dof_pos = len(dof_pos_data_transposed)
num_rows_dof_pos = (num_columns_dof_pos + 1) // 2  # 确保有足够的行来容纳所有的列

num_columns_ref_pos = len(ref_pos_data_transposed)
num_rows_ref_pos = (num_columns_ref_pos + 1) // 2  # 确保有足够的行来容纳所有的列

num_columns_actions_data = len(actions_data_transposed)
num_rows_actions_data = (num_columns_actions_data + 1) // 2  # 确保有足够的行来容纳所有的列

num_columns_motor_velocity_data = len(motor_velocity_data_transposed)
num_rows_motor_velocity_data = (num_columns_motor_velocity_data + 1) // 2  # 确保有足够的行来容纳所有的列

# 创建第一个图表，并为torque_data绘制曲线
plt.figure(figsize=(15, num_rows_torque * 4))  # 调整图表大小以容纳所有子图
for i, column in enumerate(torque_data_transposed):
    plt.subplot(num_rows_torque, 2, i+1)  # 创建子图
    plt.plot(column, label=f'Torque Curve {i+1}')
    plt.legend()
    plt.title(f'Torque Curve {i+1} - Time Series Data')

plt.suptitle('Torque Time Series Data')  # 设置总标题
plt.tight_layout()

# 创建第二个图表，并为dof_pos_data和ref_pos_data绘制曲线
plt.figure(figsize=(15, max(num_rows_dof_pos, num_rows_dof_pos) * 4))  # 调整图表大小以容纳所有子图
for i, column in enumerate(dof_pos_data_transposed):
    plt.subplot(num_rows_dof_pos, 2, i+1)  # 创建子图
    plt.plot(column, label=f'DOF Pos Curve {i+1}')
    plt.legend()
    plt.title(f'DOF Pos Curve {i+1} - Time Series Data')

for i, column in enumerate(ref_pos_data_transposed):
    plt.subplot(num_rows_ref_pos, 2, i+1)  # 创建子图
    plt.plot(column, label=f'Ref Pos Curve {i+1}')
    plt.legend()
    plt.title(f'Ref Pos Curve {i+1} - Time Series Data')

for i, column in enumerate(actions_data_transposed):
    plt.subplot(num_rows_actions_data, 2, i+1)  # 创建子图
    plt.plot(column, label=f'Action Data Curve {i+1}')
    plt.legend()
    plt.title(f'Action Data Curve {i+1} - Time Series Data')

plt.figure(figsize=(15, max(num_rows_dof_pos, num_rows_dof_pos) * 4))  # 调整图表大小以容纳所有子图
for i, column in enumerate(motor_velocity_data_transposed):
    plt.subplot(num_rows_ref_pos, 2, i+1)  # 创建子图
    plt.plot(column  , label=f'Motor Velocity Curve {i+1}'
            )
    plt.legend()
    plt.title(f'Motor Velocity Curve {i+1} - Time Series Data')

plt.suptitle('DOF Position and Reference Position Time Series Data')  # 设置总标题
plt.tight_layout()
plt.show()  # 显示第二个图表
