# Import local modules
#

from data import Data
from tfimage import TFImage
from cvimage import NuclearLayer
from matplotlib import pyplot as plt
import random

# set seed for the random number generator
#
random.seed(99)

sys_folder  = '/Volumes/X2/Projects/staging/Data/'
data_folder = sys_folder + 'data/'

# some example data
#
test = [6,7]
val = [8,9]

# call the Data class
#
d = Data(data_folder)
d.train_test_val_dir(test, val)
d.train_test_val_data()

# get random image
I, id = d.get_random_image('train')
print(I.shape)
print(id)

# Process the image
#
image = NuclearLayer(I, id)
image.preprocess(size=(512, 512), padding=0)
image.segment_embryo_image(plot_images=True)
image.border_finder(npoints = 100)
image.extend_border(inward=35, outward=-15)


# visualize the image
#
# make plots:
plt.figure(figsize=(10, 10))

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(image.I, cmap='gray')
plt.plot(image.xext, image.yext, '.', color='red', markersize=1)        # type: ignore
plt.plot(image.xint, image.yint, '.', color='green', markersize=1)      # type: ignore

# draw lines between correspoinding points of xext, yext and xint, yint using self.npoints
# for i in range(image.npoints+1):  # type: ignore
#     plt.plot([image.xext[i], image.xint[i]], [image.yext[i], image.yint[i]], linewidth=1) # type: ignore

# Plot the seglabel image
plt.subplot(1, 2, 2)
plt.imshow(image.Ilabel, cmap='gray')                           # type: ignore
for i in range(image.npoints):  # type: ignore
    plt.plot([image.xext[i], image.xint[i]], [image.yext[i], image.yint[i]], linewidth=1) # type: ignore


image.unroll(image.x, image.y)

# // plot self.Inl
plt.figure(figsize=(10, 10))
plt.imshow(image.Inl[:, :, 0], cmap='gray')
plt.show()


plt.close('all')

print("Done")
