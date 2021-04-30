# 주형오빠가 만든 디노이즈 오디오 만드려고!
# 원래위치 : data
# 사용위치 : python_import

from noise_handling import denoise_tim

load_dir = 'C:/nmb/gan/audio/listen_me_1000_22144/'
out_dir = 'C:/nmb/gan/audio/listen_me_1000_22144/denoise/'

denoise_tim(load_dir, out_dir, 5000, 15000, 512, 128, 512)

# 다 만들음!