# 오리지널이랑 불균형 둘다 만들자 그냥 ~

# 여러개의 오디오 파일 불러오기 
# 불러와서 Mel Spectogram 적용 
# 4개의 파일 *2 해서 합하기

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display

dataset = []
label = []
foldername = "imbalance_dataset2"
number = 1
# if foldername[-4] == 'f':
#     number = 0
# else : number = 1

pathAudio = 'C:/nmb/nmb_data/audio_data/' + foldername
files = librosa.util.find_files(pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:
    y, sr = librosa.load(file, sr=22050, duration=5.0)
    length = (len(y) / sr)
    if length < 5.0 : pass
    else:
        mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
        mels = librosa.amplitude_to_db(mels, ref=np.max)
        dataset.append(mels)
        label.append(number)

dataset = np.array(dataset)
label = np.array(label)
print(dataset.shape)
print(label.shape)

np.save('C:/nmb/nmb_data/imbalance_npy/'+ foldername + '_mels.npy', arr=dataset)
np.save('C:/nmb/nmb_data/imbalance_npy/' + foldername + '_label_mels.npy', arr=label)
print('=====save done=====')
# ------------------------------------------------------
# C:/nmb/nmb_data/audio_data/
# 균형 데이터
# korea_corpus_f_2m         /(456, 128, 862)
# mindslab_f_2m             /(336, 128, 862)
# open_slr_f_2m             /(912, 128, 862)
# pansori_f_2m              /(216, 128, 862)

# korea_corpus_m_2m         /(480, 128, 862)
# mindslab_m_2m             /(48, 128, 862)
# open_slr_m_2m             /(864, 128, 862)
# pansori_m_2m              /(528, 128, 862)

# 불균형 데이터
# dataset(1) : 여성 화자 20, 남성 화자 80
# imbalance_dataset1        /(480, 128, 862)

# dataset(2) : 여성 화자 80, 남성 화자 20
# imbalance_dataset2        /(480, 128, 862)


# 라벨 잘 정리되었는지 확인
# ------------------------------------------------------
'''
import numpy as np

dir = 'C:/nmb/nmb_data/0418_balance_npy/'
name = 'pansori_m_2m_label_mels'
extension = '.npy'
label = np.load(dir+name+extension)

print(np.unique(label))
'''