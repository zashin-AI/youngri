# 02에서 만들어진 npy를 오디오로 바꿔보자

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy.io.wavfile

# 저장한 npy 불러오기
<<<<<<< HEAD:gan/make_gan_03(npy to audio).py
mel = np.load('C:/nmb/gan/npy/noise1000000_total00199.npy')
=======
mel = np.load('C:/nmb/gan/npy/listen_me 1000 22144/listen_89.npy')
>>>>>>> 6d7865c51ce96e775bdf734857f107e9bb0b0f75:gan/make_gan_03_01.py
print(mel.shape)
print(mel)
mel = mel.reshape(10, 128, 173)
# 패스 지정하고 mel to audio를 저장
path = 'C:/nmb/gan/audio/'

def save_wav (wav, path):
        wav *= 32767 / max (0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write (path, 22050, wav.astype(np.int16))

for i in range(10):
    y = librosa.feature.inverse.mel_to_audio(mel[i], sr=22050, n_fft=512, hop_length=128)
<<<<<<< HEAD:gan/make_gan_03(npy to audio).py
    save_wav(y, path + 'noise1000000_total00099_' + str(i) + '.wav')
=======
    save_wav(y, path + 'listen_89_' + str(i) + '.wav')
>>>>>>> 6d7865c51ce96e775bdf734857f107e9bb0b0f75:gan/make_gan_03_01.py

print('==== audio save done ====')