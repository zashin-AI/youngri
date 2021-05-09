# 종호가 만든 오디오 자르기 참고!

from pydub import AudioSegment
import os

# ------------------------------------------------------------
# 오디오 불러오기
sumdir = 'C:/nmb/nmb_data/combine_test/F1_sum.wav'
audio = AudioSegment.from_file(sumdir)
print(type(audio))
# <class 'pydub.audio_segment.AudioSegment'>
print(audio)
# <pydub.audio_segment.AudioSegment object at 0x000002392C49A2E0>

lengaudio = len(audio)
print("len(audio)", lengaudio)
# len(audio) 216295
# 실제 오디오 길이: 3분 36초 > 216s > 216000ms
# 길이 확인

# ------------------------------------------------------------
# 임계점 설정 
start = 0
# 5s = 5000ms
threshold = 5000
end = 0
counter = 0

# 본격적인 자르기
# 시작점이 216295보다 짧으면
while start < len(audio):
    # end에 5000을 더해
    end += threshold
    print(start, end)
    # 첫번째 실행 할 때는 audio[0:5000]이겠죠
    chunk = audio[start:end]
    # file name에 0부터 차례대로 넣어서 저장
    filename = f'C:/nmb/nmb_data/F1F2F3/F1_split_5s/F1_chunk{counter}.wav'
    # 아까 자른 audio[0:5000] 부분을 내보냅니다. 위에서 정한 이름의 wav 파일로
    chunk.export(filename, format='wav')
    # 0부터 차례대로 counter를 1씩 올리고
    counter += 1
    # 시작점에도 차례대로 5000씩 더해줍시다
    start += threshold

print('==== done ====')

# 완료!
# 나머지 5초 미만은 삭제하는 걸로!