# 케라스 기본 예제 모델 말고 돌려보기~
# 기본 모델 conv1d > conv2d

# MFCCs 을 인풋으로 함!

import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/F_test_mfccs.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/F_test_label_mfccs.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/M_test_mfccs.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/M_test_label_mfccs.npy')
x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)
print(y.shape)
# (1073, 20, 862)
# (1073,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=45)
print(x_train.shape)
print(x_test.shape)
# (858, 20, 862)
# (215, 20, 862)

aaa = 2
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], int(x_train.shape[2]/aaa), aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], int(x_test.shape[2]/aaa), aaa)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# (858, 20, 431, 2)
# (215, 20, 431, 2)
# (858,)
# (215,)

# 기본 모델 conv1d > conv2d
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

    x = residual_block(inputs, 32, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 16, 3)

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
mcpath = 'C:/nmb/nmb_data/h5/another_mfcc_01.h5'
mc = ModelCheckpoint(mcpath, monitor='val_loss', verbose=1, save_best_only=True)
model.fit(x_train, y_train, epochs=128, batch_size=32, validation_split=0.2, callbacks=[stop, lr, mc])

# --------------------------------------
# 평가, 예측
model.load_weights(mcpath)

result = model.evaluate(x_test, y_test)
print('loss: ', result[0]); print('acc: ', result[1])

pred_pathAudio = 'C:/nmb/nmb_data/teamvoice_clear/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20, n_fft=512, hop_length=128)
    pred_mfccs = normalize(mfccs, axis=1)
    pred_mfccs = pred_mfccs.reshape(1, pred_mfccs.shape[0], int(pred_mfccs.shape[1]/aaa), aaa)
    y_pred = model.predict(pred_mfccs)
    # print(y_pred)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :
        print(file,(y_pred[0][0])*100,'%의 확률로 여자입니다.')
    else: print(file,(y_pred[0][1])*100,'%의 확률로 남자입니다.')

# ==========================================
# add 보다 concatenate의 val_loss가 좋지만 예측은 잘 못함 
# loss:  0.06008761376142502
# acc:  0.9767441749572754
# C:\nmb\nmb_data\teamvoice_clear\testvoice_F1(clear).wav 99.99936819076538 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_F1_high(clear).wav 99.99045133590698 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_F2(clear).wav 91.83735847473145 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_F3(clear).wav 62.20073103904724 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_M1(clear).wav 99.57851767539978 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_M2(clear).wav 99.98219609260559 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\testvoice_M2_low(clear).wav 99.9481737613678 %의 확률로 남자입니다.