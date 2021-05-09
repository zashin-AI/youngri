# 2개 해서 합쳐지는 지 확인!

import librosa
import numpy as np
import sklearn
import soundfile as sf
from pydub import AudioSegment

sounddir1 = 'C:/nmb/nmb_data/pred_voice/testvoice_F3(clear).wav'
sounddir2 = 'C:/nmb/nmb_data/pred_voice/testvoice_M1(clear).wav'

sound1 = AudioSegment.from_wav(sounddir1)
sound2 = AudioSegment.from_wav(sounddir2)
combined_sounds = sound1 + sound2
combined_sounds.export("C:/nmb/nmb_data/combine_test/test_01.wav", format="wav")

print('==== done =====')

# 제대로 합쳐지는 것 확인