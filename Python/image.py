import os
import tensorflow as tf
import cv2

class Image():
    def read_image(image_file, label):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=1,dtype=tf.uint16)
        #image = tf.image.decode_image(image,channels=1, dtype=tf.float32)
        #image = tf.image.grayscale_to_rgb(image)
        return image, label
    
    def reshapeImage(image,label):
        H     = tf.shape(image)[0]      # use tf.shape to get the height and width of the image as it is a tensor
        W     = tf.shape(image)[1]
        i0    = tf.random.uniform(shape=[1],minval=0, maxval=H-target_dim,dtype = tf.dtypes.int32)[0]
        irand = tf.random.uniform(shape=[1])[0]
        if irand<0.25:
            image = image[i0:i0+target_dim,0:target_dim,:]
        elif irand>=0.25 and irand<0.5:
            image = image[i0:i0+target_dim,H-target_dim:H,:]
        elif irand>=0.5 and irand<0.75:
            image = image[0:target_dim,i0:i0+target_dim,:]
        elif irand>=0.75:     
            image = image[W-target_dim:W,i0:i0+target_dim,:]
        return image, label
    
    def reshapeImagewBoundary(image,label,xL,yL):
        i0    = tf.random.uniform(shape=[1],minval=0, maxval=xL.shape[0],dtype = tf.dtypes.int32)[0]
        image = image[yL[i0]:yL[i0]+target_dim,xL[i0]:xL[i0]+target_dim,:]
        print(image.shape)
        return image,label,i0
    
    def reshapeNL(image,label):
        H = tf.shape(image)[0]
        W = tf.shape(image)[1]
        L = rng.uniform(shape=[1],minval=0,maxval=W-H,dtype=tf.dtypes.int32)[0]
        image  = image[:,L:L+H]
        return image,label
    
    
    