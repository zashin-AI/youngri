# 현민이가 한 방법으로 그래프 그리기 연습
# https://github.com/zashin-AI/hyunmin/blob/main/Speaker_Recognition/denoise_SVC/svc_04_%20visual.py
# 이름 project_xgb_04 으로 지정함


# 필요 라이브러리 임포트
import os
import numpy as np
import datetime
import librosa
import pickle
import warnings
from tensorflow.python.framework import device
from xgboost.core import Booster
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 시작시간 설정
str_time = datetime.datetime.now()

# 데이터 로드
x = np.load('c:/nmb/nmb_data/npy/total_data.npy')
y = np.load('c:/nmb/nmb_data/npy/total_label.npy')

x = x.reshape(-1, x.shape[1]*x.shape[2])
print(x.shape) # (4536, 110336)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 23
)
print(x_train.shape)    # (3628, 110336)
print(x_test.shape)     # (908, 110336)
print(y_train.shape)    # (3628)
print(y_test.shape)     # (908)

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델 구성 및 훈련
model = XGBClassifier(  
    learning_rate = 0.3,          # default : 0.3
    objective="binary:logistic",  # default : "binary:logistic" // "reg:squarederror"
    booster = 'gbtree',
    tree_method = 'gpu_hist',
 )

model.fit(x_train, y_train, verbose=1)

# 가중치 저장
pickle.dump(
    model,
    open(
        'c:/data/modelcheckpoint/project_xgb_04.data', 'wb')
    )

# -------------------------------------------------------------
# 그래프 그리기
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit
plt.figure(figsize=(10,6))

# # accuracy
# # train_sizes, train_scores_model, test_scores_model = \
# #     learning_curve(model, x_train[:100], y_train[:100], train_sizes=np.linspace(0.1, 1.0, 10),
# #                    scoring="accuracy", cv=7, shuffle=True, random_state=23, verbose=1)
# train_sizes, train_scores_model, test_scores_model = \
#     learning_curve(model, x_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10),
#                    scoring="accuracy", shuffle=True, random_state=23, verbose=1)


# train_scores_mean = np.mean(train_scores_model, axis=1)
# test_scores_mean = np.mean(test_scores_model, axis=1)

# # accuracy
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#                  label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#                  label="validation score")
# -----------------------------
# log loss
# train_sizes, train_scores_model, test_scores_model = \
#     learning_curve(model, x_train[:100], y_train[:100], train_sizes=np.linspace(0.1, 1.0, 10),
#                    scoring='neg_log_loss', cv=8, shuffle=True, random_state=23)
train_sizes, train_scores_model, test_scores_model = \
    learning_curve(model, x_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10),
                   scoring='neg_log_loss', shuffle=True, random_state=23)

# log loss
plt.plot(train_sizes, -train_scores_model.mean(1), 'o-', color="r", label="log_loss")
plt.plot(train_sizes, -test_scores_model.mean(1), 'o-', color="g", label="val log_loss")
# -----------------------------

plt.xlabel("Train size")
plt.ylabel("Log loss")
# plt.ylabel("Accuracy")
plt.title('XGBclassifier')
plt.legend(loc="best")

plt.show()

# -------------------------------------------------------------
# 모델 평가
y_pred = model.predict(x_test)

# 평가 지표 정의
acc_score = accuracy_score(y_test, y_pred)
log_loss = log_loss(y_test, y_pred)

print('acc : ', format(acc_score, '.4f'))
print('loss : ', format(log_loss, '.4f'))

# 모델 예측
pred_list = ['c:/nmb/nmb_data/predict/F', 'c:/nmb/nmb_data/predict/M']

count_f = 0
count_m = 0

for pred_pathAudio in pred_list:
    files = librosa.util.find_files(pred_pathAudio, ext = ['wav'])
    files = np.asarray(files)
    for file in files:
        name = os.path.basename(file)
        length = len(name)
        name = name[0]
        
        y, sr = librosa.load(file, sr = 22050)
        mels = librosa.feature.melspectrogram(
            y, sr = sr, n_fft = 512, hop_length = 128, win_length = 512
        )
        y_mel = librosa.amplitude_to_db(mels, ref = np.max)
        y_mel = y_mel.reshape(1, y_mel.shape[0] * y_mel.shape[1])

        y_mel = scaler.transform(y_mel)

        y_pred = model.predict(y_mel)

        if y_pred == 0:
            if name == 'F':
                count_f += 1
        else:
            if name == 'M':
                count_m += 1

print('43개의 여자 목소리 중 ' + str(count_f) + ' 개 정답')
print('43개의 남자 목소리 중 ' + str(count_m) + ' 개 정답')
print('time : ', datetime.datetime.now() - str_time)

# 0.9 / 2.7 / 38 / 40

# project_xgb_03
# acc :  0.920704845814978
# loss :  2.7387857792415593
# 43개의 여자 목소리 중 38 개 정답
# 43개의 남자 목소리 중 39 개 정답
# time :  0:02:12.091531

# 그래프 그리려고 돌릴 때
# time :  0:59:16.869496 ~ 1시간 10분

# -----------------------------
# 1) GPU로 돌아가게 하기 ㅇㅇ
# 2) 학원컴을 돌리기 ㄴㄴ
# 3) 그래프 그리기 (수업때 한 거) ㅇㅇ
# 4) validation loss, acc 출력하기 ㄴㄴ