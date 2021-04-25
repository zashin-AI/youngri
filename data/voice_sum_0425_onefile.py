# 그냥 flac만 wav로 바꾸려고


import librosa
from pydub import AudioSegment
import soundfile as sf
import os


pathaudio = 'D:/test/open_slr_test_m/'
save_dir = 'D:/test/open_slr_test_m/'

infiles = librosa.util.find_files(pathaudio)

for infile in infiles:
    # flac 파일의 이름 불러오기
    _, w_id = os.path.split(infile)
    # 이름에서 뒤에 .flac 떼기
    w_id = w_id[:-5]
    # flac 파일의 data, sr 불러오기
    w_data, w_sr = sf.read(infile)
    # 같은 이름으로 wav 형식으로 저장하기
    sf.write(save_dir + w_id + '.wav', w_data, w_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
print('==== flac to wav done ====')
