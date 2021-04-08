# 피피티에 넣을려고 가져옴 ㅎ

# https://keras.io/examples/audio/speaker_recognition_using_cnn/
# 참고해서 모델 만들기

# Mel-spectogram 을 인풋으로 함!

import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


title = ['MFCCs : F1_high','MFCCs : F1', 'MFCCs : F2',
          'MFCCs : F3','MFCCs : M1', 'MFCCs : M2','MFCCs : M2_low']

for i in range(7):
    mfccs = librosa.feature.mfcc(y_list[i], sr=sr_list[i])
    print('mfccs shape:', mfccs.shape)
    #(20, 216)
    
    # 정규화
    mfccs = sklearn.preprocessing.minmax_scale(mfccs, axis=1) 


# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/F_data_mel.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/F_label_mel.npy') #갓영리화이팅 영리갓 화이팅^^
m_ds = np.load('C:/nmb/nmb_data/npy/M_data_mel.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/M_label_mel.npy')
# (1073, 128, 862)
# (1073,)

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)
print(y.shape)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=45)

# input_shape
print(x_train.shape)     # (858, 128, 862)
print(x_test.shape)      # (215, 128, 862)
print(y_train.shape)     # (858,)
print(y_test.shape)      # (215,)


# 모델 구성
def residual_block(x, filters, conv_num=3, activation="relu"):

    # Conv1D에서 Maxpool1D로 반환하는 함수 정의
    s = Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding="same")(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding="same")(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPool1D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = AveragePooling1D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)

model.summary()

# 컴파일, 훈련
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=5, verbose=1)
mcpath = 'C:/nmb/nmb_data/h5/conv1_model_01_mels.h5'
mc = ModelCheckpoint(mcpath, monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=128, batch_size=32, validation_split=0.2, callbacks=[stop, lr, mc])

# --------------------------------------
# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/conv1_model_01_mels.h5')

result = model.evaluate(x_test, y_test)
print('loss: ', result[0]); print('acc: ', result[1])

pred_pathAudio = 'C:/nmb/nmb_data/teamvoice_clear/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
    y_pred = model.predict(pred_mels)
    # print(y_pred)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :
        print(file,(y_pred[0][0])*100,'%의 확률로 여자입니다.')
    else: print(file,(y_pred[0][1])*100,'%의 확률로 남자입니다.')


# tonnetz
# F1.wav 50.5% 의 확률로 여자입니다.
# F1_high.wav 50.7% 의 확률로 여자입니다.
# F2.wav 50.3% 의 확률로 여자입니다.
# F3.wav 50.5% 의 확률로 여자입니다.
# M1.wav 50.5% 의 확률로 여자입니다.
# M2.wav 50.3% 의 확률로 여자입니다.
# M2_low.wav 50.3% 의 확률로 여자입니다.

# melspectrogram
# F1.wav 100.0 %의 확률로 여자입니다.
# F1_high.wav 98.2 %의 확률로 남자입니다.
# F2.wav 99.9%의 확률로 여자입니다.
# F3.wav 99.8%의 확률로 여자입니다.
# M1.wav 99.9%의 확률로 남자입니다.
# M2.wav 99.9%의 확률로 남자입니다.
# M2_low.wav 100.0 %의 확률로 남자입니다.

# zero_crossing_rate
# F1.wav 60.0%의 확률로 남자입니다.
# F1_high.wav 60.5%의 확률로 남자입니다.
# F2.wav 60.2%의 확률로 남자입니다.
# F3.wav 60.2%의 확률로 남자입니다.
# M1.wav 60.2%의 확률로 남자입니다.
# M2.wav 60.1%의 확률로 남자입니다.
# M2_low.wav 60.3%의 확률로 남자입니다.

