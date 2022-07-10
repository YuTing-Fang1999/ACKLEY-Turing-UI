from matplotlib import pyplot as plt
import numpy as np
epoch = np.arange(0, 40)
init_value = 0.3
final_value = 1
#  step decay
# plt.plot(epoch, 1 - 0.01 * epoch)

# exponantial decay
# plt.plot(epoch, (init_value-final_value) * np.exp(-0.1 * (epoch))+final_value)

plt.plot(epoch,  ((1/40)*epoch)**np.exp(1))
plt.show()

# epoch = np.arange(0, 1, 1/40)
# print(epoch)

# epoch = np.arange(0, 40)
# print((1/40)*epoch)

# print(0.9 * np.exp(-0.2 * (epoch))[4])
