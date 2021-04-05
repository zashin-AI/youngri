# model_01 예측에 쓸 우리팀의 목소리!!

# (n), 128, 862)
# (n,)
# 의 쉐잎이 나와야 함!

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

pred_pathAudio = 'C:/nmb/data/teamvoice_clear/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
    print(pred_mels.shape)

# 쉐잎 잘 나와서 적용함~