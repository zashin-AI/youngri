import librosa
from pysndfx import AudioEffectsChain
# pip install pysndfx
import numpy as np
import math
import python_speech_features
# pip install python_speech_features
# https://github.com/jameslyons/python_speech_features
import scipy as sp
from scipy import signal

# ----------------------------------------------------------------------
# 파일 불러오기
def read_file(file_name):
    sample_file = file_name
    sample_directory = 'teamvoice/'
    sample_path = sample_directory + sample_file

    # generating audio time series and a sampling rate (int)
    x, sr = librosa.load(sample_path)

    return x, sr

x, sr = read_file("testvoice_F1.wav")

apply_audio_fx = (AudioEffectsChain()
                     .phaser()
                     .reverb())

y = apply_audio_fx(x)

# ----------------------------------------------------------------------
# 1) NOISE REDUCTION USING POWER:
    # receives an audio matrix, returns the matrix after gain reduction on noise
cent = librosa.feature.spectral_centroid(y=y, sr=sr)

threshold_h = round(np.median(cent))*1.5
threshold_l = round(np.median(cent))*0.1
print(threshold_h)
print(threshold_l)
# 1960.5
# 130.70000000000002

less_noise = AudioEffectsChain(y)#.lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)#.limiter(gain=6.0)


