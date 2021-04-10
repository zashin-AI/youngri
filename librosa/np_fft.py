# np.fft.fft 그래프 어떻게 그리는 건지 해석!!!


import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

# 목소리 불러와~
f_y, sr = librosa.load('C:/nmb/data/teamvoice_clear/testvoice_F2(clear).wav', sr=22050) 
m_y, sr = librosa.load('C:/nmb/data/teamvoice_clear/testvoice_M2(clear).wav', sr=22050)

print(len(f_y))
print('Audio length (seconds): %.2f' % (len(f_y) / sr))
print('Audio length (seconds): %.2f' % (len(m_y) / sr))
# 110250
# Audio length (seconds): 5.00
# Audio length (seconds): 5.00

# -------------------------------------------------------------------------
# 1) 2D 음파 그래프

# plt.figure(figsize=(16, 3))
# plt.subplot(1,2,1)
# plt.title('2d sound wave graph_F')
# librosa.display.waveplot(y=f_y, sr=sr)
# plt.ylabel("Amplitude")
# plt.subplot(1,2,2)
# plt.title('2d sound wave graph_M')
# librosa.display.waveplot(y=m_y, sr=sr)
# plt.ylabel("Amplitude")
# plt.show()
# ===========================
# x축: time
# y축: amplitude

# -------------------------------------------------------------------------
# 2) Fourier Transform
# 전체 오디오에 대해 x축의 시간을 주파수로 바꿔서 보자

# np.fft.fft(     # fast fourier transform 으로 1차원의 discrete fourier transform을 계산한다.
#     a           # 인풋어레이
#     ,n=None     # 아웃풋으로 나올 축의 길이 input보다 작으면 알아서 잘림, 크다면 제로패딩, 지정하지 않으면 input으로 알아서
#     ,axis=-1    # fft를 계산할 축, 지정하지 않으면 마지막으로 사용
#     ,norm=None  # { "backward", "ortho", "forward"} 정규화 모드 선택. 기본값은 backward
#     )


# 인풋어레이로 y를 넣어줌
fft = np.fft.fft(f_y)
print('fft.shape: ' , fft.shape)
print('np.max(fft): ',np.max(fft))
print('np.min(fft): ',np.min(fft))
# fft.shape: (110250,)
# np.max(fft):  (485.86803603999317-100.1009185001111j)
# np.min(fft):  (-543.1320789457668-140.21003810361378j)

# amplitude에 절대값을 씌워 모두 양수로 바꿈
amplitude = np.abs(fft) 
print('len(amplitude): ',len(amplitude))
print('np.max(amplitude): ',np.max(amplitude))
print('np.min(amplitude): ',np.min(amplitude))
# len(amplitude): 110250
# np.max(amplitude):  560.9378842304801
# np.min(amplitude):  7.990295816994493e-05

# sr=22050을 len(amplitude): 110250 만큼의 길이로 나눔
f = np.linspace(0,sr,len(amplitude))
print('len(f): ',len(f))
print('np.max(f): ',np.max(f))
print('np.min(f): ',np.min(f))
# len(f):  110250
# np.max(f):  22050.0
# np.min(f):  0.0

# y scale
left_spectrum = amplitude[:int(len(amplitude) / 2)]
print('len(left_spectrum): ',len(left_spectrum))
print('np.max(left_spectrum): ',np.max(left_spectrum))
print('np.min(left_spectrum): ',np.min(left_spectrum))
# len(left_spectrum):  55125
# np.max(left_spectrum):  560.9378842304801
# np.min(left_spectrum):  7.990295816994493e-05

# x scale
left_f = f[:int(len(amplitude) / 2)]
print('len(left_f): ',len(left_f))
print('np.max(left_f): ',np.max(left_f))
print('np.min(left_f): ',np.min(left_f))
# len(left_f):  55125
# np.max(left_f):  11024.899999092962
# np.min(left_f):  0.0

plt.figure(figsize=(10, 6))
plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("Fourier Transform")
plt.show()
