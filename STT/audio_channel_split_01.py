# pydub 에서 화자 쪼개기 가능???

from pydub import AudioSegment

# mix_audio_dir = 'C:/nmb/nmb_data/channel_split/pansori_fandm.wav'
# 채널 1이라고함
# mix_audio_dir = 'C:/nmb/nmb_data/channel_split/predict_fandm.wav'
# 채널 1이라고 함 얼탱X
mix_audio_dir = 'C:/nmb/nmb_data/channel_split/listenlistenicantlisten/youtubemix_01.wav'
# 채널 수 2라고 함 / 실제 인원 수는 9명 ㅡㅡ
# 저장한 것도 들어보면 분리 못함~

mix_audio = AudioSegment.from_file(mix_audio_dir)
print(f'원 채널 수: {mix_audio.channels}')

channelsplit = mix_audio.split_to_mono()
print(f'분리한 채널: \n female: {channelsplit[0].channels}, \n male: {channelsplit[1].channels}')

# 나눈 채널을 저장하기
channelsplit_female = channelsplit[0]
channelsplit_male = channelsplit[1]

out_dir = 'C:/nmb/nmb_data/channel_split/'
channelsplit_female.export(out_dir+'channelsplit_female.wav', format='wav')
channelsplit_male.export(out_dir+'channelsplit_male.wav', format='wav')

print('---- done -----')

# 원 채널 수: 2
# 분리한 채널:
#  female: 1,
#  male: 1