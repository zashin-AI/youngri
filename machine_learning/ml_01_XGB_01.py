# 머신 러닝 돌리자 ~
# XGBClassifier
# LGBMClassifier
# CatboostClassifier
# Randomforest

import numpy as np
import datetime 
import librosa
import sklearn
import os
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
###
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
###
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error, auc
import pickle  
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# ------------------------------------------------------------------
start_now = datetime.datetime.now()
# ------------------------------------------------------------------

model_name = 'ml_01_XGB_01_01'


# 데이터 불러오기
x_train = np.load('C:/nmb/nmb_data/npy/0428_5s_silence_denoise/project_total_npy/total_data.npy')
y_train = np.load('C:/nmb/nmb_data/npy/0428_5s_silence_denoise/project_total_npy/total_label.npy')

# print(np.unique(y_train))   > 0,1

# x_train = x_train[:100]
# y_train = y_train[:100]

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])

# print (f'x_train:{x_train.shape} x_train: {x_train.shape} \n')
# X:(4536, 110336) y: (4536,) 

# -----------------------------------------------------------------
# 전처리
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, shuffle=True, test_size = 0.2, random_state = 2021)

# print (f'x_train:{x_train.shape} y_train: {y_train.shape}')
# print (f'x_test:{x_test.shape} y_test: {y_test.shape}')
# X_train:(3628, 110336) y_train: (3628,)
# X_test:(908, 110336) y_test: (908,)

# -----------------------------------------------------------------
# 모델 구성
model = XGBClassifier(
    n_jobs = -1,
    use_label_encoder=False,
    n_estimators=10000, 
    tree_method = 'gpu_hist',
    # predictor='gpu_predictor',
    predictor='cpu_predictor',
    gpu_id=0
)
model.fit(x_train, y_train, verbose=True)#, early_stopping_rounds=10, eval_set=[(x_test, y_test)], eval_metric='logloss')

# -----------------------------------------------------------------
# model & weight save
pickle.dump(model, open('C:/nmb/nmb_data/cp/' + model_name + '.data', 'wb')) # wb : write
print("== save complete ==")

# model load
model = pickle.load(open('C:/nmb/nmb_data/cp/' + model_name + '.data', 'rb'))  # rb : read
print("== read complete ==")

# -----------------------------------------------------------------
# evaluate
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
hamm_loss = hamming_loss(y_test, y_pred)
hinge_loss = hinge_loss(y_test, y_pred)
log_loss = log_loss(y_test, y_pred)

print("hamming_loss :\t {:.4f}".format(hamm_loss))
print("hinge_loss :\t {:.4f}".format(hinge_loss))         # SVM에 적합한 cross-entropy
print("log_loss :\t {:.4f}".format(log_loss))             # Cross-entropy loss와 유사한 개념
print("accuracy :\t {:.4f}".format(accuracy))
print("recall :\t {:.4f}".format(recall))
print("precision :\t {:.4f}".format(precision))

# -----------------------------------------------------------------
# predict 데이터
pred = ['C:/nmb/nmb_data/predict/0509_5s_predict/F', 'C:/nmb/nmb_data/predict/0509_5s_predict/M']

count_f = 0
count_m = 0

for pred_pathAudio in pred:
    files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
    files = np.asarray(files)
    for file in files:
        filename = file.split('\\')[-1].split('.')[0]
        name = os.path.basename(file)
        length = len(name)
        name = name[0]
        print(name)

        y, sr = librosa.load(file, sr=22050)
        mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
        pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0]*pred_mels.shape[1])
        y_pred = model.predict(pred_mels)
        y_pred_label = np.argmax(y_pred)

        if y_pred_label == 0:   # 여성이라고 예측
            print(filename, '은 여자입니다.')
            if name == 'F' :
                count_f = count_f + 1
        else:                   # 남성이라고 예측
            print(filename, '은 남자입니다.')
            if name == 'M' :
                count_m = count_m + 1
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("42개 남성 목소리 중 "+str(count_m)+"개 정답")

# -----------------------------------------------------------------
end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)

# -----------------------------------------------------------------
'''
# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.suptitle(model_name)

plt.subplot(2, 1, 1)    # 2행 1열중 첫번째
plt.plot(history.history['loss'], marker='.', c='red', label='loss')
plt.plot(history.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 2행 1열중 두번째
plt.plot(history.history['acc'], marker='.', c='red', label='acc')
plt.plot(history.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()
'''

# ==========================================
# ml_01_XGB_01_01

