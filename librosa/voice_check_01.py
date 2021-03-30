# 여성 음성 불러오고 특성 확인을 위한 파일
# 빵형꺼 참고

# C:/nmb/data/pansori/67CV7J6E7Iic/ZBNO2Drz36c/67CV7J6E7Iic-ZBNO2Drz36c-0027.flac
# 그 딸을 너무나 사랑했을겁니다
# C:/nmb/data/pansori/7KCc6rmA7KCV/un4qbATrmx8/7KCc6rmA7KCV-un4qbATrmx8-0001.flac
# 저희는 방금 소개받은 것 처럼 의사구요 여기가 저희 진료실입니다

import librosa

# y: 소리가 떨리는 세기(진폭)를 시간 순서대로 나열한 것
# Sampling_rate(sr): 1초당 샘플의 개수, 단위 Hz 또는 kHz
y, sr = librosa.load('C:/nmb/data/pansori/7KCc6rmA7KCV/un4qbATrmx8/7KCc6rmA7KCV-un4qbATrmx8-0001.flac'
                    , sr = 22050)

print('y: ', y)
print('len(y): ', len(y))
# y:  [-0.00641101 -0.01347595 -0.01144939 ... -0.00653087 -0.00822797
#  -0.00366781]
# len(y):  110250

print('Sample rate(KHz): %d' % sr)
print('오디오의 길이(seconds): %.2f' % (len(y) / sr))
# Sample rate(KHz): 22050
# 오디오의 길이(seconds): 5.00

# ------------------------------------------------------------------------------------
# 그래프를 그려보자!
# 1) 2D 음파 그래프
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(10, 4))
plt.title('2d sound wave graph')
librosa.display.waveplot(y=y, sr=sr, max_points=50000.0, x_axis='time', offset=0.0, max_sr=1000, ax=None)
# plt.show()

# ====================================
# 2) Fourier Transform
# 시간 영역 데이터를 주파수 영역으로 변경
# y축 : 주파수(로그 스케일)
# color축 : 데시벨(진폭)

import numpy as np

D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
# np.abs: 절대값 반환 / STFT: Short-time Fourier transform / fft : fast Fourier transform
# 2048/512 = 4  의미 있는 숫자일까?
print(D.shape)
# (1025, 216)
# D: np.ndarray [shape=(1 + n_fft/2, n_frames), dtype=dtype]

plt.figure(figsize=(10, 4))
plt.title('Fourier Transform')
plt.plot(D)
# plt.show()

# ====================================
# 3) Spectogram
# 시간에 따른 신호 주파수의 스펙트럼 그래프
# 다른 이름: Sonographs, Voiceprints, Voicegrams

DB = librosa.amplitude_to_db(D, ref=np.max)
# db = decibel / amplitude_to_db : 진폭은 데시벨 스케일로 변환

plt.figure(figsize=(10, 4))
plt.title('Spectogram')
librosa.display.specshow(DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar() # 오른쪽 바 표시하려고!
# plt.show()

# ====================================
# 4) Mel Spectogram
# (인간이 이해하기 힘든) Spectogram의 y축을 Mel Scale로 변환한 것 (Non-linear transformation)
# Mel Scale: https://newsight.tistory.com/294

S = librosa.feature.melspectrogram(y, sr=sr, n_fft=2084, hop_length=512)
S_DB = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
plt.title('Mel Spectogram')
librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar()
# plt.show()

# -----------------------------------------------------------------------------------
# 오디오의 특성을 추출해보자! (Audio Feature Extraction)

# 1) Tempo (BPM: 분당 비트(Beats Per Minute) )
tempo, _ = librosa.beat.beat_track(y, sr=sr)
print('tempo: ', tempo)
# tempo:  184.5703125

# ====================================
# 3) Harmonic and Percussive Components
# Harmonics: 사람의 귀로 구분할 수 없는 특징들 (음악의 색깔)
# Percussives: 리듬과 감정을 나타내는 충격파 / Components: 요소

y_harm, y_perc = librosa.effects.hpss(y)
# hpss: Harmonic-percussive source separation

plt.figure(figsize=(10,4))
plt.title('Harmonic and Percussive Components')
plt.plot(y_harm, color='b')
plt.plot(y_perc, color='r')
plt.legend(['y_harm', 'y_perc'])
# plt.show()

# ====================================
# 6) Mel-Frequency Cepstral Coefficients (MFCCs)
# MFCCs는 특징들의 작은 집합(약 10-20)으로 스펙트럴 포곡선의 전체적인 모양을 축약하여 보여준다
# 사람의 청각 구조를 반영하여 음성 정보 추출
# https://tech.kakaoenterprise.com/66

import sklearn
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

mfccs = librosa.feature.mfcc(y, sr=sr)
mfccs = normalize(mfccs, axis=1)

print('mean: %.2f' % mfccs.mean())
print('var: %.2f' % mfccs.var())

plt.figure(figsize=(10,4))
plt.title('MFCCs')
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
# plt.show()

# ====================================
# 7) Chroma Frequencies
# 크로마 특징은 음악의 흥미롭고 강렬한 표현이다
# 크로마는 인간 청각이 옥타브 차이가 나는 주파수를 가진 두 음을 유사음으로 인지한다는 음악이론에 기반한다
# 모든 스펙트럼을 12개의 Bin으로 표현한다
# 12개의 Bin은 옥타브에서 12개의 각기 다른 반음(Semitones=Chroma)을 의미한다

chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=512)

plt.figure(figsize=(10,4))
plt.title('Chroma Frequencies')
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512)
plt.colorbar()
plt.show()