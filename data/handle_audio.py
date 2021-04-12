from pydub import AudioSegment
sound1 = AudioSegment.from_file("1.wav", format="wav")
sound2 = AudioSegment.from_file("2.wav", format="wav")

# 소리 키우기
# sound1 6 dB louder
louder = sound1 + 6

# 겹치기
# Overlay sound2 over sound1 at position 0  (use louder instead of sound1 to use the louder version)
overlay = sound1.overlay(sound2, position=0)

# 내보내기
# simple export
file_handle = overlay.export("output.mp3", format="mp3")