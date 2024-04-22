#
# This is a class to process image from the disc and extract the nuclear layer from it
#
import cv2 as cv
import numpy as np
import scipy.ndimage as nd

from functions import *


class Image:
    def __init__(self, filename):
        self.filename   = filename
        self.I          = cv.imread(filename=filename, flags=cv.IMREAD_GRAYSCALE)
        self.height_org = self.I.shape[0]
        self.width_org  = self.I.shape[1]
        
        # segmentation 
        self.Iseg   = np.empty((0, 0))
        self.Ilabel = np.empty((0, 0))
        
        # boundary
        self.npoints       = 0
        self.npoints_org   = 0
        self.inward        = 0
        self.outward       = 0
        self.depthInImage  = 0
        self.depthInEmbryo = 0
        self.x             = np.empty(0)
        self.y             = np.empty(0)
        self.xext          = self.x
        self.yext          = self.y
        self.xint          = self.x
        self.yint          = self.y
    

    def preprocess(self, size = (256, 256), padding = 22):
        """
        This method preprocesses the image by resizing it to 256x256 and adding a border of 22 pixels.
        The border is filled with the constant value of 0.
        """
        self.I      = cv.resize(self.I, size, interpolation=cv.INTER_AREA)
        self.I      = cv.copyMakeBorder(self.I, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=0)     # type: ignore
        
        # properties of the image
        self.height = self.I.shape[0]
        self.width  = self.I.shape[1]
        self.dtype  = self.I.dtype


    def segment_embryo_image(self):
        Itmp = self.I.copy()
        
        # first, we apply a Gaussian filter to smooth the image
        # Itmp = cv.GaussianBlur(Itmp, (5,5), 0)
        
        # next, we apply a morphological close operation to fill holes in the image
        se   = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
        Itmp = cv.morphologyEx(src=Itmp, op=cv.MORPH_CLOSE, kernel=se)
        
        # next, we use adpative thresholding to segment the image since the image might have uneven illumination
        Itmp = cv.adaptiveThreshold(Itmp, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,5,2)
        Itmp = cv.bitwise_not(Itmp)
        
        # next, we apply a morphological close operation to fill holes in the image
        se   = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        Itmp = cv.morphologyEx(src=Itmp, op=cv.MORPH_CLOSE, kernel=se)

        # next, we use floodfill to fill the holes in the image
        h, w   = Itmp.shape[:2]
        mask   = np.zeros((h+2, w+2), np.uint8)
        im     = Itmp.copy()
        cv.floodFill(im, mask, (0, 0), 255)             # type: ignore         
        im     = cv.bitwise_not(im)                     # invert the image
        Itmp   = Itmp | im                              # type: ignore
        
        self.Ilabel = Itmp
        self.Iseg   = np.where(Itmp == 255, self.I, 0)                              
        
     
    def border_finder(self, npoints = 100):
        self.npoints = npoints 
        
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
        x, y, _ = make_points_uniform(x, y, n = npoints)
        x       = nd.uniform_filter1d(x, size=3, mode='wrap')
        y       = nd.uniform_filter1d(y, size=3, mode='wrap')
        x, y, _ = make_points_uniform(x, y, n = npoints)
        
        self.x = x
        self.y = y
    
    
    def extend_border(self, inward = 15, outward = -5):
        # extend border outward
        self.xext, self.yext = find_normals_inward(x = self.x, y = self.y, length = outward)
        
        # extend border inward
        self.xint, self.yint = find_normals_inward(x = self.x, y = self.y, length = inward)
        
        self.inward = inward
        self.outward = outward
        self.depthInImage = np.abs(outward) + np.abs(inward)
     
    


class NuclearLayer(Image):
    def __init__(self, filename):
        super().__init__(filename)
        self.Inl = np.empty((0, 0))
    
    def unroll(self, x_nuc=None, y_nuc=None):
        if x_nuc is None or y_nuc is None:
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
        U = np.zeros((depthInImage, w, o))
        ustart = 0

        for i in range(ns):
            print(i)
            I1 = I[yL[i]:yU[i], xL[i]:xU[i], :]
            
            # plot I1
            # plt.imshow(I1, cmap='gray')
            

            # inpts = np.float32(np.column_stack((X_t1[i, :], Y_t1[i, :])))
            inpts = np.float32(np.column_stack((np.transpose(X_t1[i, :]), np.transpose(Y_t1[i, :]))))
            op = np.array([[0, 0], [ds[i], 0], [0, depthInImage], [ds[i], depthInImage]])
            outpts = np.float32(op)
            
            # print inputs and outputs
            print(inpts)
            print(outpts)

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
            plt.imshow(U[:, :, 0], cmap='gray')
        
            ustart = ufinish

        self.Inl = U
        
            
        

        

if __name__ == "__main__":
    
    # file_name = "/Volumes/X2/Projects/staging/Data/data/1_1_Embryo_1_end_of_nc12-gastrulation_czi/s1_c2_z1_t17.png"
    # file_name = "s1_c2_z1_t16.png"
    # file_name = "s5_c1_z1_t65.png"
    # file_name = "s3_c1_z1_t97.png"
    file_name = "s5_c1_z1_t150.png"
    
    from matplotlib import pyplot as plt
    print(cv.__version__)
    
    # process image
    image = NuclearLayer(file_name)
    image.preprocess(size=(512, 512), padding=88)
    image.segment_embryo_image()    
    image.border_finder(npoints = 60)
    image.extend_border(inward=15, outward=-25)
    
    # make plots:
    plt.figure(figsize=(10, 10))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image.I, cmap='gray')
    plt.plot(image.xext, image.yext, '.', color='red', markersize=1)        # type: ignore
    plt.plot(image.xint, image.yint, '.', color='green', markersize=1)      # type: ignore
    
    # draw lines between correspoinding points of xext, yext and xint, yint using self.npoints
    for i in range(image.npoints+1):  # type: ignore
        plt.plot([image.xext[i], image.xint[i]], [image.yext[i], image.yint[i]], linewidth=1) # type: ignore

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
    
    
    print("Done")