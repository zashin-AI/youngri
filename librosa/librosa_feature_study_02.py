# 각자 전담한 리브로사 피쳐 기능 공부!
# 2) librosa.feature.spectral_flatness

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
# C:/nmb/data/pansori/67CV7J6E7Iic/ZBNO2Drz36c/67CV7J6E7Iic-ZBNO2Drz36c-0027.flac
# 그 딸을 너무나 사랑했을겁니다
# C:/nmb/data/pansori/7KCc6rmA7KCV/un4qbATrmx8/7KCc6rmA7KCV-un4qbATrmx8-0001.flac
# 저희는 방금 소개받은 것 처럼 의사구요 여기가 저희 진료실입니다

y, sr = librosa.load('C:/nmb/data/pansori/7KCc6rmA7KCV/un4qbATrmx8/7KCc6rmA7KCV-un4qbATrmx8-0001.flac'
                    , sr = 22050)

print('len(y): ', len(y))
print('SR 1초당 샘플의 개수: %d' % sr)
print('오디오의 길이(초): %.2f' % (len(y)/sr))
# len(y):  110250
# SR 1초당 샘플의 개수: 22050
# 오디오의 길이(초): 5.00

# ----------------------------------------------------------------
# 2) librosa.feature.spectral_flatness
# 기능: "스펙트럼의 평탄도 계산.
# 소리가 노이즈와 얼마나 비슷한지 정량화하는 척도.
# 0~1사이의 값이 나오는데 1과 가까울 수록 백색 잡음과 유사한 것이다.

# 디폴트 값
# librosa.feature.spectral_flatness(y=None, S=None, n_fft=2048, hop_length=512,
#                                   win_length=None, window='hann', center=True,
#                                   pad_mode='reflect', amin=1e-10, power=2.0)

# 파라미터 해석
# librosa.feature.spectral_flatness(
# y=None                >> 오디오의 시간에 따른 데이터
# , S=None              >> 
# , n_fft=2048
# , hop_length=512
# , win_length=None
# , window='hann'
# , center=True
# , pad_mode='reflect'
# , amin=1e-10
# , power=2.0
# )

# ----------------------------------------------------------------
# 적용

flatness = librosa.feature.spectral_flatness(y=y)
print('원래 y.shape: ', y.shape, '\n원래 y값: \n', y)
print('flatness한 y.shape: ', flatness.shape, '\nflatness한 y값: \n', flatness)

# 원래 y.shape:  (110250,)
# flatness한 y.shape:  (1, 216)

# 원래 y값:
#  [-0.00641101 -0.01347595 -0.01144939 ... -0.00653087 -0.00822797 -0.00366781]
# flatness한 y값:
#  [[2.79141571e-02 2.69821137e-02 ... 1.28165558e-02 2.56172828e-02]]