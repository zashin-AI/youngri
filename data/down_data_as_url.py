# url로 데이터 다운하기!

import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

url = 'http://cdn.daitan.com/dataset.zip'
urllib.request.urlretrieve(url, )
local_zip = './denoise.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:/nmb/data')
zip_ref.close()

# url이 작동 안하는 걸로~ ^^