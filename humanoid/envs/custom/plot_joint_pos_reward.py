import numpy as np
import matplotlib.pyplot as plt

# 定义等价的numpy函数
def equivalent_function(diff):
    norm_diff = np.linalg.norm(diff)
    exp_term = np.exp(-1 * norm_diff)
    clamped_norm = np.minimum(0.5, norm_diff)  # 等同于torch.clamp
    linear_term = -0.2 * clamped_norm
    return -1 * exp_term -  linear_term + 0.6
diff_result = []
diff = np.linspace(-10 , 10, 100)
for i in range(0, len(diff)):
    diff_result.append(equivalent_function(diff[i]))
# 计算函数值
func_values = equivalent_function(diff)

# 绘制图像
plt.plot(diff, diff_result)
plt.title('Plot of the given function')
plt.xlabel('Index')
plt.ylabel('Function Value')
plt.show()