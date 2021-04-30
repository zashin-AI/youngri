# 원래 음성도 똑같이 시각화하자 
import matplotlib.pyplot as plt
import numpy as np

fm_ds = np.load('C:/nmb/nmb_data/npy/1s_2m_total_fm_data.npy')
fm_lb = np.load('C:/nmb/nmb_data/npy/1s_2m_total_fm_label.npy')
f_ds = fm_ds[:9600]
f_lb = fm_lb[:9600]

samples = f_ds.reshape(f_ds.shape[0], f_ds.shape[1], f_ds.shape[2])[:10]

sample_size = 10

fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

filename = 'origin_fs'

for i in range(sample_size):
    ax[i].set_axis_off()
    ax[i].imshow(samples[i])
# plt.show()

plt.savefig('C:/nmb/gan/sample/'+ filename + '.png', bbox_inches='tight')
plt.close(fig)
