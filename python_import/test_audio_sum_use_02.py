# open SLR 남성 화자만 합치자!!

import numpy as np
import librosa
from pydub import AudioSegment
import soundfile as sf
import os
from voice_handling import import_test, voice_sum, voice_split_1m

import_test()
# ==== it will be great ====

# ---------------------------------------------------------------
# voice_sum: 오디오 한 wav 파일로 합쳐서 저장하기
# def voice_sum(form, pathaudio, save_dir, out_dir):
# **** example ****
# form(파일 형식): 'wav' or 'flac'
# audio_dir(여러 오디오가 있는 파일경로) = 'C:/nmb/nmb_data/F1F2F3/F3/'
# save_dir(flac일 경우 wav파일로 저장할 경로) = 'C:/nmb/nmb_data/F1F2F3/F3_to_wave/'
# out_dir(wav파일을 합쳐서 저장할 경로+파일명까지) = "C:/nmb/nmb_data/combine_test/F3_sum.wav"

# ---------------------------------------------------------------
'''
# 폴더를 만들고 싶어유 ㅠㅠ > 만들었슈..ㅎ
dir1 = 'C:/Users/Admin/Desktop/M_wav/'
dir2 = np.arange(1, 42)
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

for i in dir2:
    createFolder( dir1 + str(i) )
'''
'''
# --------------------------------------------------------------
# 2) flac일 때
ariginal_audio_dir = 'C:/Users/Admin/Desktop/M/'
infiles = os.listdir(ariginal_audio_dir)

count = 1
for file in infiles:
    audio_dir = ariginal_audio_dir + file
    _, w_id = os.path.split(audio_dir)
    save_dir = 'C:/Users/Admin/Desktop/M_wav/'+ str(count)+'/'
    out_dir = 'C:/Users/Admin/Desktop/M_sum/'+ w_id + '.wav'
    voice_sum(form='flac', audio_dir=audio_dir, save_dir=save_dir, out_dir=out_dir)
    count += 1
# 잘 되는 것 확인!
'''

# --------------------------------------------------------------
# 적용해보자!
audio_dir = 'C:/Users/Admin/Desktop/M_sum/'
infiles = librosa.util.find_files(audio_dir)

count = 0
for file in infiles:
    origin_dir = infiles[count]
    threshold = 5000
    out_dir = 'C:/Users/Admin/Desktop/M_1m/'
    end_threshold = 60000
    voice_split_1m(origin_dir=origin_dir, threshold=threshold, end_threshold =end_threshold,out_dir=out_dir)
    count += 1

# 잘 된다잉~
