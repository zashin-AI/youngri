# 케라스 기본 예제 모델 말고 돌려보기~
# 기본 모델 conv1d > conv2d

# Mel-spectogram 을 인풋으로 함!

import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_f_mels.npy')
f_lb = np.load('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_f_label_mels.npy')
m_ds = np.load('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_m_mels.npy')
m_lb = np.load('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_m_label_mels.npy')
x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)
print(y.shape)
# (1073, 128, 862)
# (1073,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=45)
print(x_train.shape)
print(x_test.shape)
# (858, 128, 862)
# (215, 128, 862)

aaa = 2
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], int(x_train.shape[2]/aaa), aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], int(x_test.shape[2]/aaa), aaa)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# (858, 128, 431, 2)
# (215, 128, 431, 2)
# (858,)
# (215,)

# 기본 모델 conv1d > conv2d + concatenate로
# 모델 구성
model = Sequential()

def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = Conv2D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = Conv2D(filters, 3, padding="same")(x)
        x = Activation(activation)(x)
    x = Conv2D(filters, 3, padding="same")(x)
    # x = Add()([x, s])
    x= Concatenate(axis=1)([x,s])
    x = Activation(activation)(x)
    return MaxPool2D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 16, 2)
    x = residual_block(x, 8, 3)

    x = AveragePooling2D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(32, activation="relu")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)

model.summary()

# ---------------------------------------------------------------------------------------------------------
# 컴파일, 훈련
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=5, verbose=1)
mcpath = 'C:/nmb/nmb_data/h5/another_mel_01.h5'
mc = ModelCheckpoint(mcpath, monitor='val_loss', verbose=1, save_best_only=True)
# model.fit(x_train, y_train, epochs=128, batch_size=32, validation_split=0.2, callbacks=[stop, lr, mc])

# --------------------------------------
# 평가, 예측
model.load_weights(mcpath)

result = model.evaluate(x_test, y_test)
print('loss: ', result[0]); print('acc: ', result[1])

pred_pathAudio = 'C:/nmb/nmb_data/gan_audio/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], int(pred_mels.shape[1]/aaa), aaa)
    y_pred = model.predict(pred_mels)
    # print(y_pred)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :
        print(file,(y_pred[0][0])*100,'%의 확률로 여자입니다.')
    else: print(file,(y_pred[0][1])*100,'%의 확률로 남자입니다.')

# ==========================================
# loss:  0.0018979033920913935
# acc:  1.0

# C:\nmb\nmb_data\teamvoice_clear\testvoice_F1(clear).wav 99.99701976776123 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_F1_high(clear).wav 71.85341715812683 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_F2(clear).wav 99.99839067459106 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_F3(clear).wav 99.94719624519348 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_M1(clear).wav 99.99998807907104 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_M2(clear).wav 99.99996423721313 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_M2_low(clear).wav 100.0 %의 확률로 남자입니다.
# 왜케 잘 맞춰 ㅠㅠㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ