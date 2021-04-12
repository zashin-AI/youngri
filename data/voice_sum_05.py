# 여러개 합치기
# flac 파일 > wav로 바꾼 후 불러오기

import librosa
from pydub import AudioSegment
import soundfile as sf
import os

# 1) flac 오디오 wav로 바꿔 저장하기 -----------------------------------------
# 어느 폴더에서 끌어올건지 지정
who1 = 'F3/'
pathAudio1 = 'C:/nmb/nmb_data/F1F2F3/'+ who1
# util.find.files로 그 안의 모든 파일 불러오기
infiles = librosa.util.find_files(pathAudio1)
# 저장할 파일 dir 지정
save_dir = 'C:/nmb/nmb_data/F1F2F3/F3_to_wave/'

# flac to wav
for infile in infiles:
    # flac 파일의 이름 불러오기
    _, w_id = os.path.split(infile)
    # 이름에서 뒤에 .flac 떼기
    w_id = w_id[:-5]
    # flac 파일의 data, sr 불러오기
    w_data, w_sr = sf.read(infile)
    # 같은 이름으로 wav 형식으로 저장하기
    sf.write(save_dir + w_id + '.wav', w_data, w_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
            # 저장 위치                # data, sr     # 형식        # WAV에 맞는 endian     # Signed 16 bit PCM
# endian? : https://jhnyang.tistory.com/172

print('==== done(1) ====')

# 2) wav 파일 한 번에 이어서 저장하기 -----------------------------------------
who2 = 'F3_to_wave/'
pathAudio2 = 'C:/nmb/nmb_data/F1F2F3/'+ who2
infiles = librosa.util.find_files(pathAudio2)
outfile = "C:/nmb/nmb_data/combine_test/F3_sum.wav"

wavs = [AudioSegment.from_wav(wav) for wav in infiles]

combined = wavs[0]

for wav in wavs[1:]:
    combined = combined.append(wav) 

combined.export(outfile, format='wav')

print('==== done(2) ====')

# 야호!!!!!
