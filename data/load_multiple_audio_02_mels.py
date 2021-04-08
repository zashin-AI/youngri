# 여러개의 오디오 파일 불러오기 
# 불러와서 Mel Spectogram 적용 

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

dataset = []
label = []
pathAudio = 'C:/nmb/data/ForM/M/'
files = librosa.util.find_files(pathAudio, ext=['flac'])
files = np.asarray(files)
for file in files:
    y, sr = librosa.load(file, sr=22050, duration=5.0)
    length = (len(y) / sr)
    if length < 5.0 : pass
    else:
        mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
        mels = librosa.amplitude_to_db(mels, ref=np.max)

        # plt.figure(figsize=(10, 4))
        # plt.title('Mel Spectogram')
        # librosa.display.specshow(mels, sr=sr, hop_length=512, x_axis='time', y_axis='log')
        # plt.colorbar()
        # plt.show()

        dataset.append(mels)
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)
print(dataset.shape)
print(label.shape)

np.save('C:/nmb/data/npy/M_data_mel.npy', arr=dataset)
np.save('C:/nmb/data/npy/M_label_mel.npy', arr=label)
print('=====save done=====')
# ------------------------------------------------------
