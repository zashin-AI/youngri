# 각자 전담한 리브로사 피쳐 기능 공부!
# 3) librosa.feature.spectral_rolloff

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
# 3) librosa.feature.spectral_rolloff
# 기능: 컷오프할 때 생기는 roll-off(기울기)의 값
# [0 <roll_percent <1]의 값으로 가능한데, 0일 수록 가깝고 1일 수록 먼 roll-off frequency가 나온다.

# 디폴트 값
# librosa.feature.spectral_rolloff( y = None , sr = 22050 , S = None , n_fft = 2048 
                                # , hop_length = 512 , win_length = None , window = 'hann' , center = True 
                                # , pad_mode = 'reflect' , freq = None , roll_percent = 0.85 )

# 파라미터 해석
# librosa.feature.spectral_rolloff(
#     y=None              >> 오디오의 시간에 따른 데이터
# , sr=22050            >> y에 대한 오디오의 sample rate 
# , S=None              >> np.abs(librosa.stft(y)) 퓨리에 변화를 거진 y
# , n_fft=2048          >> fast Fourier transform(FFT)를 할 window 크기
# , hop_length=512      >> n_fft로 잘라 SFTF작업을 하며 겹쳐지는 길이 대체로 n_fft/4 이다.
# , win_length=None     >> window()에 의해 오디오에 만들어질 각 윈도우의 길이(대체로 n_fft와 같다.)
# , window='hann'       >> 오디오를 잘라서 변환할 때 어떤 윈도우 방법을 쓸지(scipy에서 확인)
# , center=True         >> True이면 y[t * hop_length]가 중심이 되도록 y를 패딩하고 
#                       >> False이면 y[t * hop_length]에서 시작한다.
# , pad_mode='reflect'  >> center=True이면 가장자리에 하고 default는 reflect패딩을 사용한다.
# , freq=None           >> spectrogram bins의 중심이 될 주파수
# , roll_percent=0.85   >> 롤오프 비율[0 <roll_percent <1]
# )


# ----------------------------------------------------------------
# 적용

# 스펙토그램 변환해서 적용
S, phase = librosa.magphase(librosa.stft(y))
# roll_percent=0.85 (default)로 적용
rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
print(rolloff)
# roll_percent=0.95로 적용
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
print(rolloff)
# roll_percent=0.01로 적용
rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)

# 그래프 그리기
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax)
ax.plot(librosa.times_like(rolloff), rolloff[0], label='Roll-off frequency (0.95)')
ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='w',
        label='Roll-off frequency (0.01)')
ax.legend(loc='lower right')
ax.set(title='log Power spectrogram')
plt.show()