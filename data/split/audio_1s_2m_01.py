import librosa
from pydub import AudioSegment
import soundfile as sf
import os
from voice_handling import voice_split_1m

 
origin_dir = 'D:/nmb/nmb_data/open_slr/open_slr_m_silence_split_sum_denoise/'
threshold = (1000)  # 1초씩
end_threshold = (1000*120)  # 2분
out_dir = 'D:/nmb/nmb_data/open_slr/open_slr_m_silence_split_sum_denoise_1s_2m/'

infiles = librosa.util.find_files(origin_dir)

for files in infiles :
    voice_split_1m(origin_dir=files, threshold=threshold, end_threshold =end_threshold,out_dir=out_dir)