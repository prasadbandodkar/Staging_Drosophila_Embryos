import tensorflow as tf

from sdata import SData
from cvimage import CVImage
from tfimage import TFImage

# print the version of tensorflow and whether it is using the GPU
#
print(tf.__version__)
if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")
  
  
# check if tf pipeline is working
#  
sys_folder  = '/Volumes/X2/Projects/staging/Data/'
data_folder = sys_folder + 'data/'

# some example data
#
test = [6,7]
val = [8,9]

# call the Data class
#
d = SData(data_folder)
d.train_test_val_dir(test, val)
d.train_test_val_data()


I,id = d.get_random_image('train')
image = TFImage(CVImage(I, id).image, id)

# print the shape of the image and dtype and id
#
print(image.I.shape) # type: ignore
print(image.I.dtype) # type: ignore
print(image.id)

# print max and min of the float32 tf image
#
print(tf.reduce_max(image.I))
print(tf.reduce_min(image.I))
