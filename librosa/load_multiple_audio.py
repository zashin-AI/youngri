# 여러개의 오디오 파일 불러오기 
# 불러와서 MFCC 적용 

# 여자꺼로 하는 중!!!!

import librosa
import numpy as np
import sklearn
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

dataset = []
pathAudio = 'C:/nmb/data/ForM/F/'
files = librosa.util.find_files(pathAudio, ext=['flac'])
files = np.asarray(files)
for file in files:
    y, sr = librosa.load(file, sr=16000, duration=2.0)
    print(len(y))
    length = (len(y) / sr)
    print('Audio length (seconds): %.2f' % length)
    if length < 2.0 : pass
    else:
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfccs = normalize(mfccs, axis=1)
        print(mfccs.shape)
        dataset.append(mfccs)

dataset = np.array(dataset)
print(dataset.shape)

