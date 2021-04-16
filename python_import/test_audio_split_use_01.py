# 국립 국어원 발화 말뭉치의 여성 화자를 화자별, 토픽9만 합쳐보자!!
# 합친거를 5초씩 12개 총 1분으로 나눠보자!

import librosa
from pydub import AudioSegment
import soundfile as sf
import os
from voice_handling import import_test, voice_split, voice_split_1m

import_test()
# ==== it will be great ====

# ---------------------------------------------------------------
# voice_split: 하나로 합쳐진 wav 파일을 5초씩 잘라서 dataset으로 만들기
# def voice_split(origin_dir, threshold, out_dir):
# **** example ****
# origin_dir(하나의 wav파일이 있는 경로+파일명) = 'D:/nmb_test/test_sum/test_01_wav_sum.wav'
# threshold(몇초씩 자를지 5초는 5000) = 5000
# out_dir(5초씩 잘려진 wav 파일을 저장할 경로) = 'D:/nmb_test/test_split/'

'''
# 적용해보자!
origin_dir = 'D:/nmb_test/test_sum/test_01_wav_sum.wav'
threshold = 5000
out_dir = 'D:/nmb_test/test_split/'
voice_split(origin_dir=origin_dir, threshold=threshold, out_dir=out_dir)
# 잘 된다잉~
'''

# 적용해보자!
filename = 'fz06'
origin_dir = 'C:/nmb/nmb_data/korea_corpus/korea_corpus_female/'+ filename + '.wav'
threshold = 5000
out_dir = 'C:/nmb/nmb_data/korea_corpus/korea_corpus_f_1m/'
end_threshold = 60000
voice_split_1m(origin_dir=origin_dir, threshold=threshold, end_threshold =end_threshold,out_dir=out_dir)
# 잘 된다잉~
