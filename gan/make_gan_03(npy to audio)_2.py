# 05에서 만들어진 npy를 오디오로 바꿔보자
# resize해서 문제다~~~ㅠㅠ

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy.io.wavfile

# 저장한 npy 불러오기
mel = np.load('C:/nmb/gan/npy/noise1000000_total00199.npy')
print(mel.shape)
print(mel)




'''
mel = mel.reshape(10, 128, 173)
# 패스 지정하고 mel to audio를 저장
path = 'C:/nmb/gan/audio/'

def save_wav (wav, path):
        wav *= 32767 / max (0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write (path, 22050, wav.astype(np.int16))

for i in range(10):
    y = librosa.feature.inverse.mel_to_audio(mel[i], sr=22050, n_fft=512, hop_length=128)
    save_wav(y, path + 'noise1000000_total00099_' + str(i) + '.wav')

print('==== audio save done ====')
'''