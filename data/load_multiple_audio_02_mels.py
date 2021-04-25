# 여러개의 오디오 파일 불러오기 
# 불러와서 Mel Spectogram 적용 

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display

dataset = []
label = []
<<<<<<< HEAD:data/load_multiple_audio_02_mels.py
pathAudio = 'D:/test/open_slr_test_m_denoise_silence/'
=======
pathAudio = 'D:/nmb/nmb_data/open_slr/open_slr_m_silence_split_sum_denoise_1s_2m/'
>>>>>>> 1edb7303212e106b58edc364d155d82bfecc4102:python_import/load_multiple_audio_02_mels.py
files = librosa.util.find_files(pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:
    y, sr = librosa.load(file, sr=22050, duration=1.0)
    length = (len(y) / sr)
    if length < 1.0 : pass
    else:
        mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
        mels = librosa.amplitude_to_db(mels, ref=np.max)
        dataset.append(mels)
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)
print(dataset.shape)
print(label.shape)

<<<<<<< HEAD:data/load_multiple_audio_02_mels.py
np.save('D:/test/npy/open_slr_test_m_denoise_silence_1s_2m.npy', arr=dataset)
np.save('D:/test/npy/open_slr_test_m_denoise_silence_1s_2m_label.npy', arr=label)
=======
np.save('C:/nmb/nmb_data/npy/open_slr_m_silence_split_sum_denoise_1s_2m.npy', arr=dataset)
np.save('C:/nmb/nmb_data/npy/open_slr_m_silence_split_sum_denoise_1s_2m_label.npy', arr=label)
>>>>>>> 1edb7303212e106b58edc364d155d82bfecc4102:python_import/load_multiple_audio_02_mels.py
print('=====save done=====')
# ------------------------------------------------------
