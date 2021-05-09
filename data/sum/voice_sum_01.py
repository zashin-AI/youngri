# F1,F2,F3
# 여자 2명 구분해보고 3명까지 가보자!

# 오디오 전체를 합치고 싶은데 ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅜㅠㅜㅠㅜㅠㅜㅠㅜㅠㅜ아으리이르그ㅏ > 합쳤지롱~^ㅜ^

import librosa
import numpy as np
import sklearn
import soundfile as sf

allaudio = []
dataset = []
label = []
who = 'F1/'
pathAudio = 'C:/nmb/data/F1F2F3/'+ who
files = librosa.util.find_files(pathAudio)
files = np.asarray(files)
print(len(files))
for file in files:
    y, sr = librosa.load(file, sr=22050)#, duration=5.0)
    y = np.array(y)
    print(type(y))
    allaudio.append(y)

print(type(allaudio))
print(len(allaudio))
allaudio = np.array(allaudio)
print(allaudio.shape)

np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])

# sf.write('C:/nmb/data/F1F2F3/F1_allaudio.wav', allaudio, sr)
print('=====done=====')

for file in files:
    for i in len(files):
        sound[i] = AudioSegment.from_wav(file)

'''
from pydub import AudioSegment
sound1 = AudioSegment.from_wav("filename01.wav")
sound2 = AudioSegment.from_wav("filename02.wav")
sound3 = AudioSegment.from_wav("filename03.wav")
combined_sounds = sound1 + sound2 + sound3 
combined_sounds.export("joinedFile.wav", format="wav")
'''