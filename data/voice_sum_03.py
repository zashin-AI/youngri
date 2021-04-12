# 여러개 합치기

import librosa
import numpy as np
import sklearn
import soundfile as sf
import wave

who = 'F1/'
pathAudio = 'C:/nmb/nmb_data/F1F2F3/'+ who
infiles = librosa.util.find_files(pathAudio)
outfile = "C:/nmb/nmb_data/combine_test/test_03.wav"

data = []
for infile in infiles:
    w = wave.open(infile, 'rb')
    data.append([w.getparams(), w.readframes(w.getnframes())])
    w.close()

output = wave.open(outfile, 'wb')
output.setparams(data[0][0])
output.writeframes(data[0][1])
output.writeframes(data[1][1])
output.close()

print('==== done ====')

# 안됩니다유?ㅎ