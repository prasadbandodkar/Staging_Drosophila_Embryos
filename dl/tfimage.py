#
# This is a class to host tf related image processing functions
#

import tensorflow as tf
import numpy as np

class TFImage:
    def __init__(self, image, id):
        self.I = tf.convert_to_tensor(np.array(image), dtype=tf.uint8)
        self.id = id
        
    def augment(self, seed):
        image = tf.image.stateless_random_flip_left_right(self.I, seed=seed)
        image = tf.image.stateless_random_flip_up_down(self.I, seed=seed)
        image = tf.image.stateless_random_brightness(self.I, max_delta=0.5, seed=seed)
        image = tf.image.stateless_random_contrast(self.I, lower=0.5, upper=1.5, seed=seed)
        return image
    
    def normalize(self):
        # make sure image pixels are in the range [min, max]
        #
        imin = tf.math.reduce_min(self.I)
        imax = tf.math.reduce_max(self.I)
        image = (self.I - imin)/(imax-imin)
        return image

    def make_tf_data(self, list_type):
        pass
    