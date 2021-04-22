# 스피치 레코그나이션 사용하기

from hanspell import spell_checker
import kss

# ---------------------------------------------------------------
import os
import librosa
import speech_recognition as sr
r= sr.Recognizer()

audios_dir = 'C:/nmb/nmb_data/STT voice/'
infiles = librosa.util.find_files(audios_dir)

for audio in infiles:
    # w_id로 이름 불러오기
    _, w_id = os.path.split(audio)
    w_id = w_id[:-4]
    test = sr.AudioFile(audio)
    # 오디오 데이터로 인식시키기
    # <class 'speech_recognition.AudioFile'> -> <class 'speech_recognition.AudioData'>
    with test as source:
        recog_audio = r.record(source)
    # recognition 에 돌리기
    result = r.recognize_google(recog_audio, language='ko-KR')
    # 파일명이랑 결과 반환
    print(w_id, ': ', result)



'''
    # 맞춤법 검사를 돌리면 어떨까?
    spelled_sent = spell_checker.check(result)
    checked_sent = spelled_sent.checked
    # print(w_id, ': \n', result); print(w_id + '_spell: \n', checked_sent)

    # 문장 분리해서 편하게 확인하기
    print(w_id, ':')
    for sent in kss.split_sentences(checked_sent): print(sent)
'''