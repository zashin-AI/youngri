import cv2
import numpy as np
from numpy.core.fromnumeric import resize


X = np.load('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_f_mels.npy')
print(X.shape)

x = np.resize(X[0], (28,28))
print(x.shape)
'''
resize_list =[]
for x in X:
    x = np.resize(x, (28,28))
    resize_list.append(x)
resize_list = np.array(resize_list)
print(resize_list.shape)

# np.save('C:/nmb/nmb_data/0418_balance_denoise_npy/resize_f_mels.npy', arr=resize_list)

# ------------------------------------
# 원본 이미지 저장

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ImageHelper(object):
    def save_image(self, generated, epoch, directory):
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(generated[count], cmap='gray')
                axs[i,j].axis('off')
                count += 1
        fig.savefig("{}/{}_3.png".format(directory, epoch))
        plt.close()

save_img = ImageHelper()
# save_img.save_image(resize_list[50:75], 10, 'C:/nmb/gan/make_gan_05/original_img')

wide = np.load('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_f_mels.npy')
save_img.save_image(wide[50:75], 10, 'C:/nmb/gan/make_gan_04/original_img')
'''