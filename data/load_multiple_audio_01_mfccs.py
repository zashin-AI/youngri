# 여러개의 오디오 파일 불러오기 
# 불러와서 MFCC 적용 

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

dataset = []
label = []
pathAudio = 'C:/nmb/data/ForM/F/'
files = librosa.util.find_files(pathAudio, ext=['flac'])
files = np.asarray(files)
for file in files:
    y, sr = librosa.load(file, sr=22050, duration=5.0)
    length = (len(y) / sr)
    if length < 5.0 : pass
    else:
        mfccs = librosa.feature.mfcc(y, sr=sr, n_fft=512, hop_length=128, n_mfcc=20)
        mfccs = normalize(mfccs, axis=1)
        print('mfccs: ', mfccs.shape)
        dataset.append(mfccs)
        label.append(0)

dataset = np.array(dataset)
label = np.array(label)
print(dataset.shape) # (545, 20, 862)
print(label.shape) # (545,)

np.save('C:/nmb/data/npy/F_test_mfccs.npy', arr=dataset)
np.save('C:/nmb/data/npy/F_test_label_mfccs.npy', arr=label)
# print('=====save done=====')
# ------------------------------------------------------
# F_mfccs
# (545, 20, 216)
# (545,)

# M_mfccs
# (528, 20, 216)
# (528,)
