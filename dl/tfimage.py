#
# This is a class to host tf related image processing functions
#

import tensorflow as tf
import numpy as np

class TFImage:

    def __init__(self, image, id):
        # Convert the uint8 image to a TensorFlow tensor directly
        self.I = tf.convert_to_tensor(np.array(image), dtype=tf.uint8)
        self.I = tf.image.convert_image_dtype(self.I, dtype=tf.float32) # convert to float32
        self.id = id
        
    def augment(self, seed):
        seed1, seed2 = seed, (seed[0], seed[1] + 1)
        image = tf.image.stateless_random_flip_left_right(self.I, seed=seed)
        image = tf.image.stateless_random_flip_up_down(image, seed=seed2)
        
        seed2 = (seed2[0], seed2[1] + 1)
        image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=seed2)
        
        seed2 = (seed2[0], seed2[1] + 1)
        image = tf.image.stateless_random_contrast(image, lower=0.5, upper=1.5, seed=seed2)
        return image
    
    
    def normalize(self):
        # make sure image pixels are in the range [min, max]
        #
        imin = tf.math.reduce_min(self.I)
        imax = tf.math.reduce_max(self.I)
        image = (self.I - imin)/(imax-imin)
        return image

    