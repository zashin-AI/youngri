# 각자 전담한 리브로사 피쳐 기능 공부!

import librosa
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

# ----------------------------------------------------------------
# 1) spectral_contrast
# 기능: 

# librosa.feature.spectral_contrast(
#       y=None
#     , sr=22050
#     , S=None
#     , n_fft=2048
#     , hop_length=512
#     , win_length=None
#     , window='hann'
#     , center=True
#     , pad_mode='reflect'
#     ,freq=None
#     , fmin=200.0
#     , n_bands=6
#     , quantile=0.02
#     , linear=False
#     )

librosa.feature.spectral_contrast(
    y=y
    , sr=22050
    , S=None
    , n_fft=2048
    , hop_length=512
    , win_length=None
    , window='hann'
    , center=True
    , pad_mode='reflect'
    ,freq=None
    , fmin=200.0
    , n_bands=6
    , quantile=0.02
    , linear=False
    )

