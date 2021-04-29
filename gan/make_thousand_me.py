# 내 목소리 1000개 만들어서 그걸로 돌리기

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display

file = 'C:/nmb/nmb_data/teamvoice_clear/testvoice_F2(clear).wav'

y, sr = librosa.load(file, sr=22050, duration=1.0)
mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
mels = librosa.amplitude_to_db(mels, ref=np.max)

methousand = []
for i in range(1000):
    methousand.append(mels)

methousand = np.array(methousand)
print(methousand.shape)

np.save('C:/nmb/nmb_data/npy/me_thousand.npy', arr = methousand)
'''
dataset = np.array(dataset)
label = np.array(label)
print(dataset.shape)
print(label.shape)
'''