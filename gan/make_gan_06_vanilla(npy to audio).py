# 06에서 만들어진 npy를 audio로 바꿔 들어보자!

import tensorflow as tf
import numpy as np
import librosa
import scipy.io.wavfile

# 저장한 npy 불러오기
mel = np.load('C:/nmb/gan_0504/npy/b300_e1000_n500/b300_e1000_n500_total00010.npy')
print(mel.shape)
# (24, 128, 173)

# 패스 지정하고 mel to audio를 저장
path = 'C:/nmb/gan_0504/audio/b300_e1000_n500/'

def save_wav (wav, path):
        wav *= 32767 / max (0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write (path, 22050, wav.astype(np.int16))

for i in range(mel.shape[0]):
    y = librosa.feature.inverse.mel_to_audio(mel[i], sr=22050, n_fft=512, hop_length=128)
    save_wav(y, path + 'b300_e1000_n500_total10_' + str(i) + '.wav')

print('==== audio save done ====')