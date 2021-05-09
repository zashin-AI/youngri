# 주형오빠가 만든 디노이즈 오디오 만드려고!
# 원래위치 : data
# 사용위치 : python_import

from noise_handling import denoise_tim
import os

load_dir = 'C:/nmb/gan_0504/audio/b300_e1000_n500/'
ext_dir = 'C:/nmb/gan_0504/audio/b300_e1000_n500/'

denoise_tim(load_dir, ext_dir, 0, 110250, 512, 128, 512)

