import librosa
from pydub import AudioSegment
import soundfile as sf
import os
from voice_handling import import_test, voice_split, voice_split_1m

def voice_split_1m(origin_dir, threshold, end_threshold, out_dir):
    audio = AudioSegment.from_file(origin_dir)
    _, w_id = os.path.split(origin_dir)
    w_id = w_id[:-4]
    lengaudio = len(audio)
    # 임계점 설정(1s = 1000ms)
    start = 0
    threshold = threshold
    end = 0
    counter = 0
    end_threshold = end_threshold
    # 본격적인 잘라서 저장하기
    while start < end_threshold:
        end += threshold
        print(start, end)
        chunk = audio[start:end]
        filename = out_dir + w_id + f'_{counter}.wav'
        chunk.export(filename, format='wav')
        counter += 1
        start += threshold
    print('==== wav split done ====')

 
origin_dir = 'D:/nmb/1,3,5,7_dataset/1s_24=120s/korea_corpus_f_split_sum/'
threshold = (10000)  # 10초씩
end_threshold = (1000*120)  # 2분
out_dir = 'D:/nmb/1,3,5,7_dataset/1s_24=120s/korea_corpus_f_10s_2m/'

infiles = librosa.util.find_files(origin_dir)

for files in infiles :
    voice_split_1m(origin_dir=files, threshold=threshold, end_threshold =end_threshold,out_dir=out_dir)