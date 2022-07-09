from matplotlib import pyplot as plt
import numpy as np
epoch = np.arange(0, 40)
init_value = 0.3
final_value = 1
#  step decay
# plt.plot(epoch, 1 - 0.01 * epoch)

# exponantial decay
plt.plot(epoch, (init_value-final_value) * np.exp(-0.1 * (epoch))+final_value)
plt.show()

# print(0.9 * np.exp(-0.2 * (epoch))[4])
