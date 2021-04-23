import sys
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence   
import librosa
from voice_handling import import_test, voice_sum, split_silence


# 집컴에서!
'''
# 여러 오디오('wav')가 있는 파일경로
audio_dir = 'C:/nmb/nmb_data/korea_corpus/korea_corpus_f_sum/'
# 묵음 부분 마다 자른 오디오 파일을 저장할 파일 경로(이 경로안에 새로운 파일을 만들어준다)
split_silence_dir = 'C:/nmb/nmb_data/korea_corpus/korea_corpus_f_slience_split/'
# 묵음 부분 마다 자른 오디오 파일을 합쳐서 저장할 파일경로
sum_dir = 'C:/nmb/nmb_data/korea_corpus/korea_corpus_f_slience_split_sum/'

split_silence(audio_dir=audio_dir, split_silence_dir=split_silence_dir, sum_dir=sum_dir)

# korea_corpus 완료
'''

# 학원에서!
