from re import L
from flask import Flask, request, render_template, send_file
# Flask: 웹 구동을 위한 부분
# request: 파일을 업로드 할 때 flask 서버에서 요청할 때 쓰는 부분
# render_template: html을 불러올 때 필요한 부분
# send_file: 파일을 flask 서버에 보낼 때 쓰는 부분

from tensorflow.keras.models import load_model
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence

import numpy as np
import librosa
import speech_recognition as sr
import tensorflow as tf
import os
import copy

# 필요함수 정의
r = sr.Recognizer()

def normalized_sound(audio_file):
    audio = AudioSegment.from_wav(audio_file)
    normalizedsound = effects.normalize(audio)
    return normalizedsound

def split_silence(audio_file):
    dbfs = audio_file.dBFS
    audio_chunks = split_on_silence(
        audio_file,
        min_silence_len=1000,
        silence_thresh=dbfs - 30,
        keep_silence=True
    )
    return audio_chunks

def STT(audio_file):
    with audio_file as audio:
        file = r.record(audio)
        stt = r.recognize_google(file, language='ko-KR')
    return stt

def predict_speaker(y, sr):
    mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512, win_length=512)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])

    y_pred = model.predict(pred_mels)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :
        return '여자'
    if y_pred_label == 1 :
        return '남자'

app = Flask(__name__)

# 첫 화면 (파일 업로드)
@app.route('/')

def upload_file():
    return render_template('upload.html')

# 업로드 후에 출력 되는 화면 (추론)
@app.route('/uploadFile', methods = ['POST'])
def download():
    # 파일이 업로드 되면 실시 할 과정
    if request.method == 'POST':
        f = request.files['file']
        if not f : return render_template('upload.html')

        f_path = os.path.splitext(str(f))
        f_path = os.path.split(f_path[0])

        folder_path = 'c:/nmb/nmb_data/web/chunk/'
        f_name = f_path[1]

        normalizedsound = normalized_sound(f)
        audio_chunks = split_silence(normalizedsound)
        len_audio_chunks = len(audio_chunks)

        save_script = ''

        for i, chunk in enumerate(audio_chunks):
            speaker_stt = list()
            out_file = folder_path + '/' + str(i) + '_chunk.wav'
            chunk.export(out_file, format = 'wav')
            aaa = sr.AudioFile(out_file)

            try:
                stt_text = STT(aaa)
                speaker_stt.append(str(stt_text))

                y, sample_rate = librosa.load(out_file, sr = 22050)

                if len(y) >= 22050 * 5:
                    y= y[:22050*5]
                    speaker = predict_speaker(y, sample_rate)
                    speaker_stt.append(str(speaker))
                    print(speaker_stt[1], " : ", speaker_stt[0])

                else:
                    audio_copy = AudioSegment.from_wav(out_file)
                    audio_copy = copy.deepcopy(audio_copy)  # deepcopy를 안하고 그냥 카피하면 복사본이랑 원본이 연결되어 있기때문에 이 기능을 쓴다.
                    for num in range(3):
                        audio_copy = audio_copy.append(copy.deepcopy(audio_copy), crossfade=0)
                    audio_copy.export(folder_path + '/' + str(i) + '_chunks_over_5s.wav', format='wav')
                    y_copy, sample_rate = librosa.load(folder_path + '/' + str(i) + '_chunks_over_5s.wav')
                    y_copy  = y_copy[:22050*5]
                    speaker = predict_speaker(y_copy, sample_rate)
                    speaker_stt.append(str(speaker))
                    print(speaker_stt[1] + ' : ' + speaker_stt[0])

                save_script += speaker_stt[1] + ' : ' + speaker_stt[0] + '\n\n'
                with open('c:/nmb/nada/web/static/test.txt', 'wt', encoding='utf-8') as f : f.writhlines(save_script)
            
            except:
                pass
        return render_template('/download.html')

# 파일 다운로드
@app.route('/download/')
def download_file():
    file_name = 'c:/nmb/nada/web/static/test.txt'
    return send_file(
        file_name,
        as_attachment = True,   # as_attachment = False 일 경우 첨부가 안 되어서 다운 안됨
        mimetype = 'text/txt',
        cache_timeout = 0 # 0으로 지정해서 초기화하지 않으면 이전 파일이 다운 받아짐
    )

# 추론 된 파일 읽기
@app.route('/read')
def read_text():
    f = open('c:/nmb/nada/web/static/test.txt', 'r', encoding = 'utf-8')
    return '</br>'.join(f.readlines())

if __name__ == '__main__' :
    model = load_model('c:/data/modelcheckpoint/mobilenet_rmsprop_1.h5')
    app.run(debug=True) # False 인 경우 어떤 문제인지 출력해주지 않는다.





        


