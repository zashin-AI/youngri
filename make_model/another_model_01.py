# 케라스 기본 예제 모델 말고 돌려보기~

# Mel-spectogram 을 인풋으로 함!

import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/pansori_2s_F_test_mels.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/pansori_2s_F_test_label_mels.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/pansori_2s_M_test_mels.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/pansori_2s_M_test_label_mels.npy')
x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)
print(y.shape)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=45)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
