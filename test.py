import numpy as np

# diff_IQM = np.array([-0.8,0.1])
diff_IQM = np.array([[-0.5,0.1], [-0.8,0.1]])
target_diff_IQM = np.array([-0.5,0.07])
weight = np.array([0.5,0.5])
priority = (diff_IQM/target_diff_IQM).dot(weight)

print(priority<1.4)
diff_IQM = diff_IQM[priority<1.4]
priority = priority[priority<1.4]
idx = np.argmax(priority)
print(priority)
print(idx)
print(diff_IQM[idx])