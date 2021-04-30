import numpy as np
import matplotlib.pyplot as plt
import random

sample = np.load('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_f_mels.npy')
print(sample.shape)
print(np.max(sample))

a = random.randint(1, 1920)
sam1 = sample[a]
sam1 = sam1.reshape(128, 862, 1)
print(sam1.shape)

plt.imshow(sam1)
plt.show()


# n = 128
# for i in range(1, n+1):
#     if n%i == 0:
#         print(i)
