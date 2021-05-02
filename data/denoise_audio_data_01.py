# 주형오빠가 만든 디노이즈 오디오 만드려고!
# 원래위치 : data
# 사용위치 : python_import

from noise_handling import denoise_tim
import os

load_dir = 'C:/Users/lemon/OneDrive/바탕 화면/fm_5s_denoise/t18/'
ext_dir = 'C:/Users/lemon/OneDrive/바탕 화면/fm_5s_denoise/'

denoise_tim(load_dir, ext_dir, 5000, 15000, 512, 128, 512)

