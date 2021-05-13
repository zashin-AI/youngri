# 06에서 만들어진 npy를 audio로 바꿔 들어보자!

import tensorflow as tf
import numpy as np
import librosa
import scipy.io.wavfile

# 저장한 npy 불러오기
<<<<<<< HEAD
mel = np.load('C:/nmb/gan_0504/npy/b96_e10000_n100/b96_e10000_n100_total10000.npy')
=======
mel = np.load('C:/nmb/gan_0504/npy/b100_e5000_n100_male/b100_e5000_n100_male_total05000.npy')
>>>>>>> 27a2e9746f969d30ff34658f0932877f900b077f
print(mel.shape)
# (24, 128, 173)

print(np.max(mel))
print(np.min(mel))

# 패스 지정하고 mel to audio를 저장
<<<<<<< HEAD
path = 'C:/nmb/gan_0504/audio/b96_e10000_n100_total10000/'
=======
path = 'C:/nmb/gan_0504/audio/b100_e5000_n100_male/'
>>>>>>> 27a2e9746f969d30ff34658f0932877f900b077f

def save_wav (wav, path):
        wav *= 32767 / max (0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write (path, 22050, wav.astype(np.int16))

for i in range(mel.shape[0]):
    y = librosa.feature.inverse.mel_to_audio(mel[i], sr=22050, n_fft=512, hop_length=128)
<<<<<<< HEAD
    save_wav(y, path + 'b96_e10000_n100_total10000_' + str(i) + '.wav')
=======
    save_wav(y, path + 'b100_e5000_n100_male_total05000' + str(i) + '.wav')
>>>>>>> 27a2e9746f969d30ff34658f0932877f900b077f

print('==== audio save done ====')
