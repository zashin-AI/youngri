from itertools import count
import numpy as np
import os
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate, LeakyReLU, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop
from keras import backend as K

start_now = datetime.now()

# 데이터 불러오기
f_ds = np.load('D:/nmb/0602_10s/total_10s/10s_f_mels.npy')
f_lb = np.load('D:/nmb/0602_10s/total_10s/10s_f_mels_label.npy')
m_ds = np.load('D:/nmb/0602_10s/total_10s/10s_m_mels.npy')
m_lb = np.load('D:/nmb/0602_10s/total_10s/10s_m_mels_label.npy')
x = np.concatenate([f_ds, m_ds], 0);y = np.concatenate([f_lb, m_lb], 0)
print(x.shape, y.shape)
# (1918, 128, 1723) (1918,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

aaa = 1 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)
print(x_train.shape, y_train.shape) # (3628, 128, 862, 1) (3628,) > (1534, 128, 1723, 1) (1534,)
print(x_test.shape, y_test.shape)   # (908, 128, 862, 1) (908,) > (384, 128, 1723, 1) (384,)


model = MobileNet(
    include_top=True,
    input_shape=(128,1723,1),
    classes=2,
    pooling=None,
    weights=None,
)

model.summary()
# model.trainable = False

# 지표 정의하기
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# model.save('C:/data/modelcheckpoint/mobilenet_rmsprop_10s_model.h5')

# 컴파일, 훈련
op = RMSprop(lr=1e-3)
batch_size = 8

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/data/modelcheckpoint/mobilenet_rmsprop_10s.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)

model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc', f1_m, recall_m, precision_m])
history = model.fit(x_train, y_train, epochs=1000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
model.load_weights('C:/data/modelcheckpoint/mobilenet_rmsprop_10s.h5')
result = model.evaluate(x_test, y_test, batch_size=8)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))
print("f1_score : {:.5f}".format(result[2]))
print("recall_m : {:.5f}".format(result[3]))
print("precision_m : {:.5f}".format(result[4]))


############################################ PREDICT ####################################

pred = ['D:/nmb/0602_10s/predict_10/10s_F', 'D:/nmb/0602_10s/predict_10/10s_M']

count_f = 0
count_m = 0

for pred_pathAudio in pred:
    files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
    files = np.asarray(files)
    for file in files:
        name = os.path.basename(file)
        length = len(name)
        name = name[0]

        y, sr = librosa.load(file, sr=22050)
        mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
        pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
        y_pred = model.predict(pred_mels)
        y_pred_label = np.argmax(y_pred)
        if y_pred_label == 0:   # 여성이라고 예측
            print(file, '{:.4f} 의 확률로 여자입니다.'.format((y_pred[0][0])*100))
            if name == 'F' :
                count_f = count_f + 1
        else:                   # 남성이라고 예측
            print(file, '{:.4f} 의 확률로 남자입니다.'.format((y_pred[0][1])*100))
            if name == 'M' :
                count_m = count_m + 1
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("43개 남성 목소리 중 "+str(count_m)+"개 정답")

end = datetime.now()
time = end - start_now
print("작업 시간 : ", time)
