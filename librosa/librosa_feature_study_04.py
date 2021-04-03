# 각자 전담한 리브로사 피쳐 기능 공부!
# 4) librosa.feature.tonnetz

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from librosa.feature import spectral_contrast
from librosa.feature import spectral_flatness
from librosa.feature import spectral_rolloff
from librosa.feature import poly_features

# ----------------------------------------------------------------
# 음성 불러오기

filename = 'testvoice'
filegender = '_F2'
filetype = '.wav'
sample_directory = 'C:/nmb/data/teamvoice/'
sample_path = sample_directory + filename + filegender + filetype

y, sr = librosa.load(sample_path, sr=22050)

print('len(y): ', len(y))
print('SR 1초당 샘플의 개수: %d' % sr)
print('오디오의 길이(초): %.2f' % (len(y)/sr))
# len(y):  110250
# SR 1초당 샘플의 개수: 22050
# 오디오의 길이(초): 5.00

# ----------------------------------------------------------------
# 4) librosa.feature.tonnetz
# 기능: 음색 중심의 특성을 계산(tonnetz)
# chroma features를 이용해 완전5도, 단3도 ,장3도를 각각 2 차원 좌표로 보여준다.
# 음색 3개 * x,y-axis = 행을 6개로 나누어서 출력함.

# 디폴트 값
# librosa.feature.tonnetz(y=None, sr=22050, chroma=None, **kwargs

# 파라미터 해석
# librosa.feature.tonnetz(
#     y=None       >> 오디오의 시간에 따른 데이터
# , sr=22050       >> y에 대한 오디오의 sample rate 
# , chroma=None    >> 각 프레임에을 크로마 피쳐 뽑은 값, 없을 경우 cqt chromagram을 수행한다.
# , **kwargs       >> 크로마가 미리 계산 되지 않았을 때 chroma_cqt에 대한 추가 키워드
# ) 

# ---------------------------------------------------------------
# 적용

# y에서 음색 성분을 해서 tonnetz에 적용
y = librosa.effects.harmonic(y)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
print('tonnetz: ', tonnetz)
# tonnetz:  [[-0.12201151 -0.08998461 -0.06241944 ... -0.03429811 -0.05752611
#   -0.08172323]...
#    [-0.05382726 -0.0535487  -0.04232097 ... -0.01774389 -0.01651231
#   -0.00094176]]

# chroma_cqt 값과 tonnetz feature 비교하기
fig, ax = plt.subplots(nrows=2, sharex=True)
img1 = librosa.display.specshow(tonnetz,
                                y_axis='tonnetz', x_axis='time', ax=ax[0])
ax[0].set(title='Tonal Centroids (Tonnetz)'+'_'+ filegender)
ax[0].label_outer()
img2 = librosa.display.specshow(librosa.feature.chroma_cqt(y, sr=sr),
                                y_axis='chroma', x_axis='time', ax=ax[1])
ax[1].set(title='Chroma')
fig.colorbar(img1, ax=[ax[0]])
fig.colorbar(img2, ax=[ax[1]])
plt.show()