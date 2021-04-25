import librosa
from pydub import AudioSegment
import soundfile as sf
import os
from voice_handling import voice_split_1m

 
<<<<<<< HEAD:data/audio_1s_2m_01.py
origin_dir = 'D:/test/open_slr_test_m_denoise/'
threshold = (1000)  # 1초씩
end_threshold = (1000*120)  # 2분
out_dir = 'D:/test/open_slr_test_m_denoise_silence/'
=======
origin_dir = 'D:/nmb/nmb_data/open_slr/open_slr_m_silence_split_sum_denoise/'
threshold = (1000)  # 1초씩
end_threshold = (1000*120)  # 2분
out_dir = 'D:/nmb/nmb_data/open_slr/open_slr_m_silence_split_sum_denoise_1s_2m/'
>>>>>>> 1edb7303212e106b58edc364d155d82bfecc4102:python_import/audio_1s_2m_01.py

infiles = librosa.util.find_files(origin_dir)

for files in infiles :
    voice_split_1m(origin_dir=files, threshold=threshold, end_threshold =end_threshold,out_dir=out_dir)