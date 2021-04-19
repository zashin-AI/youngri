# 스피치 레코그나이션 사용하기

import speech_recognition as sr
import librosa
import os
from hanspell import spell_checker

# ---------------------------------------------------------------
# 인식 기능 불러오기
r= sr.Recognizer()

# 여러 오디오 있는 경로 지정
audios_dir = 'C:/Users/lemon/OneDrive/바탕 화면/STT voice/'
infiles = librosa.util.find_files(audios_dir)

# 여러 wav 파일 하나씩 돌리기
for audio in infiles:
    # 이름 불러오기
    _, w_id = os.path.split(audio)
    w_id = w_id[:-4]
    # 지정된 경로의 오디오 파일 불러오기
    test = sr.AudioFile(audio)
    # 오디오 데이터로 인식시키기
    with test as source:
        recog_audio = r.record(source)
    # recognition 에 돌리기
    result = r.recognize_google(recog_audio, language='ko-KR')
    # 파일명이랑 결과 반환
    # print(w_id, ': ', result)

    # 맞춤법 검사를 돌리면 어떨까?
    spelled_sent = spell_checker.check(result)
    checked_sent = spelled_sent.checked
    print(w_id, ': \n', result); print(w_id + '_spell: \n', checked_sent)

# test_01 : 
#  하루 확진자가 오늘로 나흘째 600명 때 머물고 있습니다 하지만 이스라엘은 사실상 집단면역 선언하고 오늘부터 야외에서 마스크를 벗고 있습니다 JTBC 취재팀이 직접 이스라엘로 날아갔는데 잠시 후 상지 연결해 보겠습니다
# test_01_spell: 
#  하루 확진자가 오늘로 나흘째 600명 때 머물고 있습니다 하지만 이스라엘은 사실상 집단면역 선언하고 오늘부터 야외에서 마스크를 벗고 있습니다 JTBC 취재팀이 직접 이스라엘로 날아갔는데 잠시 후 상지 연결해 보겠습니다
# test_02 : 
#  9 토끼와 자라 옛날에 어느 바다 속에 아주 아름다운 용궁이 있었어요 그런데이 아름다운 용궁에 슬픈 일이 생겼답니다 나이 많은 용왕님이 시름시름 앓다가 자리에 누워 있기 때문이지요
# test_02_spell: 
#  9 토끼와 자라 옛날에 어느 바닷속에 아주 아름다운 용궁이 있었어요 그런데 이 아름다운 용궁에 슬픈 일이 생겼답니다 나이 많은 용왕님이 시름시름 앓다가 자리에 누워 있기 때문이지요
# test_F1 : 
#  사람 간이 m 이상 거리두기 거리두기 지침에 따라이 자리는 비워 주소 
# test_F1_spell: 
#  사람 간이 m 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주소
# test_F2 : 
#  실내 사람 간이 m 이상 거리두기 거리두기 지침에 따라이 자리는 비워 주세요
# test_F2_spell:
#  실내 사람 간이 m 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주세요
# test_F3 : 
#  사람 간이 m 이상 거리두기 거리두기 치면 따라이 자리는 비워 주소
# test_F3_spell:
#  사람 간이 m 이상 거리두기 거리두기 치면 따라 이 자리는 비워 주소
# test_F4 :
#  실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라이 자리는 비어 주세요
# test_F4_spell:
#  실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비어 주세요
# test_M1 :
#  실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 자리는 비워 주세요
# test_M1_spell:
#  실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 자리는 비워 주세요
# test_M2 :
#  술래 가람관 2 m 이상 거리보기 지침에 따라이 자리는 비워 줘
# test_M2_spell:
#  술래 가람과 2 m 이상 거리 보기 지침에 따라 이 자리는 비워 줘
# test_M3 :
#  실내 사람 간이 m 이상 거리두기 거리두기 지침에 따라이 자리는 비워 주세요
# test_M3_spell:
#  실내 사람 간이 m 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주세요
# test_M4 :
#  실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라이 자리는 비워 주세요
# test_M4_spell:
#  실내 사람과 님이 더 이상 거리두기 거리두기 지침에 따라 이 자리는 비워 주세요