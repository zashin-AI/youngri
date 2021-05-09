# 종호가 만든 오디오 자르기 참고!

from pydub import AudioSegment
import os

sumdir = 'C:/nmb/nmb_data/combine_test/F1_sum.wav'
audio = AudioSegment.from_file(sumdir)
lengthaudio = len(audio)
print("len(audio)", lengthaudio)
# len(audio) 216295

start = 0
# 5s = 5000ms
threshold = 5000
end = 0
counter = 0

while start < len(audio):
    end += threshold
    print(start, end)
    chunk = audio[start:end]
    filename = f'C:/nmb/nmb_data/F1F2F3/F1_split_5s/F1_chunk{counter}.wav'
    chunk.export(filename, format='wav')
    counter += 1
    start += threshold

print('==== done ====')

# 완료!
# 나머지 5초 미만은 삭제하는 걸로!