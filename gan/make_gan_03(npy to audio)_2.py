# 05에서 만들어진 npy를 오디오로 바꿔보자
# resize해서 문제다~~~ㅠㅠ

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy.io.wavfile

# 저장한 npy 불러오기

# 원하는 쉐잎 = 110250

DCgan = np.load('C:/nmb/gan/make_gan_05/npy/DC_n22144,e10000,b100_09900.npy')
Dcgan = DCgan.reshape(100, 100, 100)

resize_list =[]
for wave in Dcgan:
    wave = np.resize(wave, (128, 862))
    resize_list.append(wave)
resize_list = np.array(resize_list)
print(resize_list.shape)


# 패스 지정하고 mel to audio를 저장
path = 'C:/nmb/gan/make_gan_05/audio/'

def save_wav (wav, path):
        wav *= 32767 / max (0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write (path, 22050, wav.astype(np.int16))

for i in range(10):
    y = librosa.feature.inverse.mel_to_audio(resize_list[i], sr=22050, n_fft=512, hop_length=128)
    save_wav(y, path + 'DC_n22144,e10000,b100_09900_' + str(i) + '.wav')

print('==== audio save done ====')
