# 각자 전담한 리브로사 피쳐 기능 공부!
# 1) spectral_contrast

import numpy as np
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

filename = 'testvoice'
filegender = '_M2'
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
# 2d 음파 그래프 먼저 확인!
import matplotlib.pyplot as plt
import librosa.display

# plt.figure(figsize=(10,4))
# plt.title('2D sound wave graph')
# librosa.display.waveplot(y=y, sr=sr)
# plt.show()

# ----------------------------------------------------------------
# 1) spectral_contrast
# 기능: 퓨리에 변환을 거친 주파수를 band 수만큼 잘라 음의 대비를 구한다.
# 높은 대비값은 깨끗하고 정확한 소리이고 낮은 대비값일 수록 노이즈이다.
# 각 행은 주어진 옥타브 기반 주파수에 해당

# 디폴트 값
# librosa.feature.spectral_contrast(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
#                                   win_length=None, window='hann', center=True, pad_mode='reflect',
#                                   freq=None, fmin=200.0, n_bands=6, quantile=0.02, linear=False)

# 파라미터 해석
# librosa.feature.spectral_contrast(
#     y=None              >> 오디오의 시간에 따른 데이터
# , sr=22050            >> y에 대한 오디오의 sample rate
# , S=None              >> np.abs(librosa.stft(y)) 퓨리에 변화를 거진 y
# , n_fft=2048          >> fast Fourier transform(FFT)를 할 window 크기
# , hop_length=512      >> n_fft로 잘라 SFTF작업을 하며 겹쳐지는 길이 대체로 n_fft/4 이다.
# , win_length=None     >> window()에 의해 오디오에 만들어질 각 윈도우의 길이(대체로 n_fft와 같다.)
# , window='hann'       >> 오디오를 잘라서 변환할 때 어떤 윈도우 방법을 쓸지(scipy에서 확인)
# , center=True         >> True이면 y[t * hop_length]가 중심이 되도록 y를 패딩하고 
#                         >> False이면 y[t * hop_length]에서 시작한다.
# , pad_mode='reflect'  >> center=True이면 가장자리에 하고 default는 reflect패딩을 사용한다.
# ,freq=None            >> spectrogram bins의 중심이 될 주파수
# , fmin=200.0          >> 컷오프를 할 주파수의 값 (20kHz와 관련있어보임)
# , n_bands=6           >> 주파수를 나눌 band의 수(y축보면 확인)
# , quantile=0.02       >> 파장의 마루(꼭대기)와 골(계곡)을 결정하기 위한 분위수
# , linear=False        >> True일 경우 선형차이를 반환(peaks - valleys), False일 경우 로그차이를 반환(log(peaks) - log(valleys))
# )

# ----------------------------------------------------------------
# 적용

# Short-time Fourier transform 으로 변환
S = np.abs(librosa.stft(y))
# 특성 추출 적용
contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

# 그래프로 그리자
fig, ax = plt.subplots(nrows=2, sharex=True)
img1 = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[0])
fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
ax[0].set(title='Power Spectrogram'+'_'+ filegender)
ax[0].label_outer()
img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1])
fig.colorbar(img2, ax=[ax[1]])
ax[1].set(ylabel='Frequency bands', title='Spectral contrast'+'_'+ filegender)
plt.show()

