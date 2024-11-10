#
# This is a class to process image from the disc and extract the nuclear layer from it
#
import cv2 as cv
import numpy as np
import scipy.ndimage as nd
from skimage import measure
import matplotlib.pyplot as plt
import torch

from Python.functions import *


class EmbryoImage:
    def __init__(self, I, id, **kwargs):
        
        # Inputs
        #
        # image
        self.I          = I
        self.id         = id
        # size and padding are used for preprocessing. First, the image is resized to size
        # and then a border of padding pixels is added to all sides of the image.   
        #
        self.size          = kwargs.get('size', (512, 512))
        self.padding       = kwargs.get('padding', 44)
   
        # plot_images is a boolean to plot images during processing
        self.plot_images   = kwargs.get('plot_images', False)

        # npoints is the number of points to use for the border
        self.npoints       = kwargs.get('npoints', 100)
       
        # inward and outward are the lengths to extend the border inward and outward
        self.inward        = kwargs.get('inward', 40)
        self.outward       = kwargs.get('outward', -24)
        
        # Outputs
        #
        self.height_org = self.I.shape[0]
        self.width_org  = self.I.shape[1]
        
        # segmentation 
        self.Iseg   = np.empty((0, 0))
        self.Ilabel = np.empty((0, 0))
               
        # outputs
        self.npoints_org   = 0      # from original segmentation
        self.depthInImage  = np.abs(self.outward) + np.abs(self.inward)
        self.x             = np.empty(0)
        self.y             = np.empty(0)
        self.xext          = self.x
        self.yext          = self.y
        self.xint          = self.x
        self.yint          = self.y
    

    def preprocess(self):
        """
        This method preprocesses the image by resizing it to 256x256 and adding a border of 22 pixels.
        The border is filled with the constant value of 0.
        """
        self.I      = cv.resize(self.I, self.size, interpolation=cv.INTER_AREA)
        self.I      = cv.copyMakeBorder(self.I, self.padding, self.padding, self.padding, self.padding, cv.BORDER_REPLICATE)     # type: ignore
        
        # properties of the image
        self.height = self.I.shape[0]
        self.width  = self.I.shape[1]
        self.dtype  = self.I.dtype


    def segment_embryo_image_old(self):
        Itmp = self.I.copy()
        
        # first, we apply a Gaussian filter to smooth the image
        plt.subplot(2, 4, 1)
        plt.imshow(Itmp, cmap='gray')
        plt.title('After Gaussian Blur')

        # next, we apply a morphological close operation to fill holes in the image
        se   = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
        Itmp = cv.morphologyEx(src=Itmp, op=cv.MORPH_CLOSE, kernel=se)
        plt.subplot(2, 4, 2)
        plt.imshow(Itmp, cmap='gray')
        plt.title('After Morphological Closing')

        # next, we use adaptive thresholding to segment the image since the image might have uneven illumination
        Itmp = cv.adaptiveThreshold(src = Itmp, 
                                    maxValue=255, 
                                    adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    thresholdType=cv.THRESH_BINARY_INV,
                                    blockSize=13,
                                    C=-1)                       # type: ignore
        plt.subplot(2, 4, 3)
        plt.imshow(Itmp, cmap='gray')
        plt.title('After Adaptive Thresholding')

        # Apply a morphological close operation
        Itmp = cv.morphologyEx(src=Itmp, op=cv.MORPH_CLOSE, kernel=se)
        plt.subplot(2, 4, 4)
        plt.imshow(Itmp, cmap='gray')
        plt.title('After First Morphological Closing')

        # Apply another morphological close operation
        Itmp = cv.morphologyEx(src=Itmp, op=cv.MORPH_CLOSE, kernel=se)
        plt.subplot(2, 4, 5)
        plt.imshow(Itmp, cmap='gray')
        plt.title('After Second Morphological Closing')

        # Apply a third morphological close operation
        Itmp = cv.morphologyEx(src=Itmp, op=cv.MORPH_CLOSE, kernel=se)
        plt.subplot(2, 4, 6)
        plt.imshow(Itmp, cmap='gray')
        plt.title('After Third Morphological Closing')

        # Invert the image since the object of interest is whiter than the background
        Itmp = cv.bitwise_not(Itmp)
        plt.subplot(2, 4, 7)
        plt.imshow(Itmp, cmap='gray')
        plt.title('After Inverting')

        # next, we use floodfill to fill the holes in the image
        h, w   = Itmp.shape[:2]
        mask   = np.zeros((h+2, w+2), np.uint8)
        im     = Itmp.copy()
        cv.floodFill(im, mask, (0, 0), 255)             # type: ignore         
        im     = cv.bitwise_not(im)                     # invert the image
        Itmp   = Itmp | im                              # type: ignore
        plt.subplot(2, 4, 8)
        plt.imshow(Itmp, cmap='gray')
        plt.title('After Flood Fill')

        self.Ilabel = Itmp
        self.Iseg   = np.where(Itmp == 255, self.I, 0)
        # plt.subplot(2, 4, 9)
        # plt.imshow(self.Ilabel, cmap='gray')
        # plt.title('Final Segmented Image')

        plt.tight_layout()
        plt.show()
        # print('done')
  
    
    def segment_embryo_image(self):
        Itmp = self.I.copy()
        
        # Normalize the image between its max and min values
        Itmp = cv.normalize(Itmp, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)    # type: ignore
        
        # gaussian blur
        Itmp = cv.GaussianBlur(Itmp, (9, 9), 0)
        
        # open and close to remove noise
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        Itmp = cv.morphologyEx(Itmp, cv.MORPH_OPEN, se)     # type: ignore
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        Itmp = cv.morphologyEx(Itmp, cv.MORPH_CLOSE, se)     # type: ignore
        
        if self.plot_images:
            plt.subplot(2, 4, 1)
            plt.imshow(Itmp, cmap='gray')
            plt.title('After Open and Closing')
        
        
        # Next, binarize the image using Adaptive Thresholding
        Itmp = cv.adaptiveThreshold(Itmp, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        # Next, binarize the image using regular Thresholding
        # _, Itmp = cv.threshold(Itmp, 128, 255, cv.THRESH_BINARY)
        Itmp = cv.bitwise_not(Itmp) 

        if self.plot_images:
            plt.subplot(2, 4, 3)
            plt.imshow(Itmp, cmap='gray')
            plt.title('After Adaptive Thresholding')

        # Add a close operation to fill holes in the image
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (33, 33))
        Itmp = cv.morphologyEx(Itmp, cv.MORPH_CLOSE, se)     # type: ignore

        if self.plot_images:
            plt.subplot(2, 4, 4)
            plt.imshow(Itmp, cmap='gray')
            plt.title('After Closing')

        # Add another close operation to fill holes in the image
        # se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (32, 32))
        # Itmp = cv.morphologyEx(Itmp, cv.MORPH_CLOSE, se)     # type: ignore

        # if plot_images:
        #     plt.subplot(2, 4, 5)
        #     plt.imshow(Itmp, cmap='gray')
        #     plt.title('After Closing, Opening 2')

        # Next, flood fill the image. We need to invert the image first
        h, w = Itmp.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv.floodFill(Itmp, mask, (0, 0), 255)     # type: ignore

        if self.plot_images:
            plt.subplot(2, 4, 6)
            plt.imshow(mask, cmap='gray')
            plt.title('Mask: After Flood Fill')

        Itmp = mask
        Itmp = cv.bitwise_not(Itmp)

        # Now, find the biggest connected component in the image
        labels = measure.label(Itmp, connectivity=2)
        unique, counts = np.unique(labels, return_counts=True) # type: ignore
        unique = unique[1:]
        counts = counts[1:]
        max_label = unique[np.argmax(counts)]
        Itmp = np.where(labels == max_label, 255, 0)

        if self.plot_images:
            plt.subplot(2, 4, 7)
            plt.imshow(Itmp, cmap='gray')
            plt.title('After Finding Biggest Connected Component')

        if self.plot_images:
            plt.tight_layout()
            plt.show()
        
        self.Ilabel = Itmp

        # print('done')
        
        
    def border_finder(self):
        
        # Convert the image to CV_8UC1
        self.Ilabel = cv.convertScaleAbs(self.Ilabel)
        # first, we find all the contours of the segmented image
        contours = cv.findContours(image=self.Ilabel, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)[0]            # type: ignore

        # next, we find the contour with the maximum area
        max_area = 0
        for i, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # extract x and y from max_contour
        x = max_contour[:, :, 0].squeeze()
        y = max_contour[:, :, 1].squeeze()
        self.npoints_org = x.shape[0]
        
        # now, we need to find the boundary such that each x,y is unique, the number of points is npoints,
        # and the points are uniformly distributed along the contour
        #
        # procedure:
        # 1. make the points uniform using npoints_org - length of the contour
        # 2. smooth the points using a uniform filter with a wrap mode. Use 10% of the number of points
        # 3. make the points uniform again
        # 4. smooth the points using a uniform filter with a wrap mode. Use 3% of the number of points.
        # 5. make the points uniform again
        #
        x, y, _   = make_points_uniform(x, y, n = self.npoints_org)
        nsmooth = int(0.10 * max_contour.shape[0])                      # 5% of the number of points
        x       = nd.uniform_filter1d(x, size=nsmooth, mode='wrap')
        y       = nd.uniform_filter1d(y, size=nsmooth, mode='wrap')
        x, y, _ = make_points_uniform(x, y, n = self.npoints)
        x       = nd.uniform_filter1d(x, size=3, mode='wrap')
        y       = nd.uniform_filter1d(y, size=3, mode='wrap')
        x, y, _ = make_points_uniform(x, y, n = self.npoints)
        
        self.x = x
        self.y = y
    
    
    def extend_border(self):
        # extend border outward
        self.xext, self.yext = find_normals_inward(x = self.x, y = self.y, length = self.outward)
        
        # extend border inward
        self.xint, self.yint = find_normals_inward(x = self.x, y = self.y, length = self.inward)
        
     
    

class NuclearLayer(EmbryoImage):
    def __init__(self, I, id, **kwargs):
        super().__init__(I, id, **kwargs)
        self.Inl = np.empty((0, 0))
        self.unroll()
    
    def unroll(self):
        if self.xext.size == 0 or self.yext.size == 0:
            self.preprocess()
            self.segment_embryo_image()
            self.border_finder()
            self.extend_border()
            x_nuc = self.xext
            y_nuc = self.yext
        else:
            x_nuc = self.xext
            y_nuc = self.yext

        depthInImage = self.depthInImage
        
        I = self.I.copy()

        I = cv.morphologyEx(I, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20)))
        
        # make I 3D
        I = I[:, :, np.newaxis]

        m, n, o = I.shape
        ns = len(self.xext) - 1
        x_im = np.arange(n)
        y_im = np.arange(m)

        xext = roundx(self.xext, x_im)[0]
        yext = roundx(self.yext, y_im)[0]
        xint = roundx(self.xint, x_im)[0]
        yint = roundx(self.yint, y_im)[0]
        x_nuc = roundx(x_nuc, x_im)[0]
        y_nuc = roundx(y_nuc, y_im)[0]

        X_t = np.column_stack((xext[:-1], xext[1:], xint[:-1], xint[1:]))
        Y_t = np.column_stack((yext[:-1], yext[1:], yint[:-1], yint[1:]))

        xL = np.floor(np.min(X_t, axis=1)).astype(int)
        yL = np.floor(np.min(Y_t, axis=1)).astype(int)
        xU = np.ceil(np.max(X_t, axis=1)).astype(int)
        yU = np.ceil(np.max(Y_t, axis=1)).astype(int)

        X_t1 = X_t - xL[:, None] + 1
        Y_t1 = Y_t - yL[:, None] + 1

        dx = np.diff(x_nuc, axis=0)
        dy = np.diff(y_nuc, axis=0)
        ds = np.round(np.sqrt(dx**2 + dy**2)).astype(int)
        w = np.sum(ds)
        U = np.zeros((depthInImage, w, o), dtype=np.uint8)
        ustart = 0

        for i in range(ns):

            I1 = I[yL[i]:yU[i], xL[i]:xU[i], :]
            
            # plot I1
            # plt.imshow(I1, cmap='gray')
            

            # inpts = np.float32(np.column_stack((X_t1[i, :], Y_t1[i, :])))
            inpts = np.float32(np.column_stack((np.transpose(X_t1[i, :]), np.transpose(Y_t1[i, :]))))
            op = np.array([[0, 0], [ds[i], 0], [0, depthInImage], [ds[i], depthInImage]])
            outpts = np.float32(op)

            M = cv.getPerspectiveTransform(inpts, outpts)       # type: ignore

            # Define the size of the output image (width, height)
            # size = (int(np.max(op[:, 0])), int(np.max(op[:, 1])))
            size = (int(op[1, 0]), int(op[2, 1]))
            It = cv.warpPerspective(I1, M, size, flags=cv.INTER_LINEAR)       # type: ignore

            ufinish = ustart + ds[i]
            # transpose It. It is 2d array
            # It = np.transpose(It)
            U[:, ustart:ufinish, 0] = It
            
            # show U
            # plt.imshow(U[:, :, 0], cmap='gray')
        
            ustart = ufinish
        
        self.Inl = U
        



class CVImage(NuclearLayer):
    def __init__(self, I, id, **kwargs):
        super().__init__(I, id, **kwargs)
        self.trunc_width = kwargs.get('trunc_width', None)
        self.image  = self.get_unrolled_image()
        
    
    def get_unrolled_image(self):
        Inl = self.Inl
        
        if self.trunc_width is not None:
            max_start = Inl.shape[1] - self.trunc_width
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            image = Inl[:, start:start + self.trunc_width, :]
        else:
            image = Inl
        
        return image

        
        
        
        
        
                

if __name__ == "__main__":
    
    print("Hello from CVImage")