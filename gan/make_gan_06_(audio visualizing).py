# https://teddylee777.github.io/tensorflow/vanilla-gan

# 1점대로 다루기 여간 어려우니...2점대로 만들자

import numpy as np
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --------------------------------------------------------------------------
# 데이터 불러오기
fm_ds = np.load('C:/nmb/nmb_data/npy/1s_2m_total_fm_data.npy')
f_ds = fm_ds[:9600]

f_ds = f_ds.reshape(f_ds.shape[0], f_ds.shape[1]*f_ds.shape[2])
print(f_ds.shape)
# (9600, 22144)
# ------------------------------------------------------------------------
# Normalize
# generator 마지막에 activation이 tanh.
# tanh을 거친 output 값이 -1~1 사이로 나오기 때문에 최대 1 최소 -1 로 맞춰줘야 한다.

print(np.max(f_ds), np.min(f_ds))
# 3.8146973e-06 -80.0

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
scaler1 = StandardScaler()
scaler1.fit(f_ds)
f_ds = scaler1.transform(f_ds)

scaler2 = MaxAbsScaler()
scaler2.fit(f_ds)
f_ds = scaler2.transform(f_ds)

# 이 값이 -1 ~ 1 사아에 있는지 확인
print(np.max(f_ds), np.min(f_ds))
# 1.0 -0.9439434
# 비슷하게 맞춰 줌!

# ------------------------------------------------------------------------

generated_images = f_ds.reshape(9600, 128, 173)[:24]

filename = 'f_ds_original'

plt.figure(figsize=(8, 4))
for i in range(generated_images.shape[0]):
    plt.subplot(4, 6, i+1)
    plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
    plt.axis('off')
plt.tight_layout()
# plt.show()
plt.savefig('C:/nmb/gan_0504/visualize/'+ filename + '.png', bbox_inches='tight')


print('==== done ====')