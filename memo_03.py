# 종호가 안된다고 해서 해보는 파일~

from pydub import AudioSegment

audio = AudioSegment.from_('C:/nmb/nmb_data/ForM/F/106_003_0199.flac', format='flac')
lengthaudio = len(audio)
print("Length of Audio File", lengthaudio)

# 종호야 미안~ 나도 안 돼 ^^