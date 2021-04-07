import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display



y, sr = librosa.load('C:/nmb/data/pansori/7KCV6riw7KCV/GJu8ZETMTZU/7KCV6riw7KCV-GJu8ZETMTZU-0036.flac'\
                    , sr=22050)

'''
mfccs = librosa.feature.mfcc(y, sr=sr, n_fft=512, hop_length=128, center=False)
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
mfccs = normalize(mfccs, axis=1)

print(mfccs.shape)
'''

mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128, n_mels=100, center=False)
mels = librosa.amplitude_to_db(mels, ref=np.max)

print(mels.shape)


# (20, 862)
# 이게 원래 쉐이프입니다!!!
# 아래가 center = False입니다!!!

# (20, 858)
# 여러분 우리가! 답을 찾았습니다~!!!!! 키워드는 center 랑 loose frame 입니다..!
# 설명은 현민아 해줄겁니다!!!
# 그전에 원래 계산공식은 n_frames = 1 + int((len(y)-n_fft)/hop_length) 입니다!!!!!!
# 근데!!!!! 윈도우 사이즈랑 프레임가 정확하게 동일하지 않기 때문에 마지막에 몇개의 프레임을 잃습니다..! 저런!
# 그래서 center=Flase를 하면 ....어쨌든 해결이 됩니다!

'''
print('len(y):', len(y))
# len(y): 110250
n_fft = 512
hop = 128

n_frames = 1 + int((len(y)-n_fft)/hop)

print(n_frames)
'''