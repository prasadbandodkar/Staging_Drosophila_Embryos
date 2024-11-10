# Imports
#
from sdata import SData
from cvimage import CVImage
from matplotlib import pyplot as plt    # type: ignore
import random

# set values
random.seed(4)
test = [6,7]
val  = [8,9]


# get data
sys_folder  = '/Volumes/X2/Projects/staging/Data/'
data_folder = sys_folder + 'data/'
d = SData(data_folder, test, val)


# get random image
I, id, folder, idx = d.get_random_image('test')
print(I.shape)
print(id)
print(folder)
print(idx)


# create image
size        = (512, 512)
padding     = 44
plot_images = True
npoints     = 100
inward      = 40
outward     = -24
image = CVImage(I, id, size, padding, plot_images, npoints, inward, outward)



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


# print image dtype
print(image.Inl.dtype)

# // plot self.Inl
# make folder new window
plt.figure(figsize=(10, 10))
plt.imshow(image.Inl[:, :, 0], cmap='gray')
plt.show()

# plt.close('all')

print("Done")
