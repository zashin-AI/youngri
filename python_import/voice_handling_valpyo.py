# 오디오 합치기와 나누기를 함수로 정의해보자!

import librosa
from pydub import AudioSegment
import soundfile as sf
import os

# 테스트 한번 불러서 출력하고 가져가세요~
def import_test():
    print('==== it will be great ====')

# ---------------------------------------------------------------
# voice_sum: 오디오를 한 wav 파일로 합쳐서 저장하기

# example
# form(파일 형식): 'wav' or 'flac'
# audio_dir(여러 오디오가 있는 파일경로)
# save_dir(flac일 경우 wav파일로 저장할 경로)
# out_dir(wav파일을 합쳐서 저장할 경로+파일명)

def voice_sum(form, audio_dir, save_dir, out_dir):
    if form =='flac':
        infiles = librosa.util.find_files(audio_dir)
        for infile in infiles:
            _, w_id = os.path.split(infile)
            w_id = w_id[:-5]
            w_data, w_sr = sf.read(infile)
            sf.write(save_dir + w_id + '.wav', w_data, w_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
        print('==== flac to wav done ====')
        infiles = librosa.util.find_files(save_dir)
        wavs = [AudioSegment.from_wav(wav) for wav in infiles]
        combined = wavs[0]
        for wav in wavs[1:]:
            combined = combined.append(wav) 
        combined.export(out_dir, format='wav')
        print('==== wav sum done ====')

    if form == 'wav':
        infiles = librosa.util.find_files(audio_dir)
        wavs = [AudioSegment.from_wav(wav) for wav in infiles]
        combined = wavs[0]
        for wav in wavs[1:]:
            combined = combined.append(wav) 
        combined.export(out_dir, format='wav')
        print('==== wav sum done ====')


# ---------------------------------------------------------------
# voice_split: 하나로 합쳐진 wav 파일을 5초씩 잘라서 dataset으로 만들기

# **** example ****

# origin_dir(하나의 wav파일이 있는 경로+파일명)
# threshold(몇초씩 자를지 5초는 5000) = 5000
# out_dir(5초씩 잘려진 wav 파일을 저장할 경로)


def voice_split(origin_dir, threshold, out_dir):
    audio = AudioSegment.from_file(origin_dir)
    _, w_id = os.path.split(origin_dir)
    w_id = w_id[:-4]
    lengaudio = len(audio)
    start = 0
    threshold = threshold
    end = 0
    counter = 0
    while start < len(audio):
        end += threshold
        print(start, end)
        chunk = audio[start:end]
        filename = out_dir + w_id + f'{counter}.wav'
        chunk.export(filename, format='wav')
        counter += 1
        start += threshold
    print('==== wav split done ====')
