# 여러개 합치기

import librosa
from pydub import AudioSegment

# 누구의 폴더인지 지정
who = 'F1/'
pathAudio = 'C:/nmb/nmb_data/F1F2F3/'+ who
# util.find.files로 그 안의 모든 파일 불러오기
infiles = librosa.util.find_files(pathAudio)
# 저장할 파일 dir 지정
outfile = "C:/nmb/nmb_data/combine_test/F1_sum.wav"

# infiles 안의 wav 불러오기 for문 압축
wavs = [AudioSegment.from_wav(wav) for wav in infiles]
# 가장 앞에 시작될 오디오 지정
combined = wavs[0]
# 그 다음 오디오 부터 위의 combined에 붙이기
for wav in wavs[1:]:
    combined = combined.append(wav) 
# 모두 합친 파일을 wav로 내보내기
combined.export(outfile, format='wav')

print('==== done ====')

# 성공!! 후레이!!