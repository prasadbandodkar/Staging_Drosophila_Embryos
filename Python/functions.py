import os
import numpy as np
import torch


def add_noise(data, unique_count, noise_range=(1e-3, 1e-2)):
    """Add noise to data if not all elements are unique."""
    if len(np.unique(data)) < unique_count:
        noise = noise_range[0] + np.random.rand(unique_count) * (noise_range[1] - noise_range[0])
        data = data + noise
    return data
    
    
    
def make_points_uniform(x, y, n):
    """
    Make the points of a contour uniform.
    
    Parameters:
    x (np.array): x-coordinates of the points
    y (np.array): y-coordinates of the points
    n (int): number of points for the new contour
    
    Returns:
    tuple: new x-coordinates, new y-coordinates, and new cumulative pseudo arclength
    """
    # Perimeter of the contour
    dx        = np.diff(x)
    dy        = np.diff(y)
    ds        = np.sqrt(dx**2 + dy**2)
    perimeter = np.sum(ds)
    arclen    = np.concatenate(([0], np.cumsum(ds)))  # cumulative pseudo arclength for original perim

    # Changing the density of points around the contour
    n = n + 1 if n % 2 else n
    s = np.concatenate(([0], np.cumsum(perimeter/n*np.ones(n))))  # cumulative pseudo arclength for new perim

    # add noise to the points so that they are not all unique
    point_count = len(x)
    x           = add_noise(x, point_count)
    y           = add_noise(y, point_count)
    arclen      = add_noise(arclen, point_count)
    arclen[0]   = 0

    # Interpolating the points
    x2     = np.interp(s, arclen, x)
    y2     = np.interp(s, arclen, y)
    x2[-1] = x2[0]
    y2[-1] = y2[0]

    # Perimeter of the new contour
    dx            = np.diff(x2)
    dy            = np.diff(y2)
    ds            = np.sqrt(dx**2 + dy**2)
    new_perimeter = np.sum(ds)
    arclen2       = 2*np.concatenate(([0], np.cumsum(ds)))/new_perimeter - 1    # cumulative pseudo arclength for new perim

    return x2, y2, arclen2



def find_normals_inward(x, y, length = 5):
    """
    Finds points locally normal to x, y, that are Yhatmax closer to xc, yc.

    Args:
        x, y: numpy arrays representing the x and y coordinates of the points.
        Yhatmax: the depth into the embryo that we keep. Default, 5 pxl. A negative value will extend the border outward.

    Returns:
        x2, y2: numpy arrays representing the x and y coordinates of the new points.
    """

    # initialize variables
    ns = len(x) - 1
    x2 = np.zeros(ns + 1)
    y2 = np.zeros(ns + 1)

    # add noise to the points
    x = x + np.random.uniform(0.01, 0.09, ns + 1)
    y = y + np.random.uniform(0.01, 0.09, ns + 1)

    # find the normals
    a1               = np.roll(x, -1)[:-1] - np.roll(x, 1)[:-1]
    a2               = np.roll(y, -1)[:-1] - np.roll(y, 1)[:-1]
    b2               = a1 / (a1**2 + a2**2)
    b1               = -(a2 / a1) * b2
    b1[np.isnan(b1)] = 0
    b2[np.isnan(b2)] = 0
    d                = np.sqrt(b1**2 + b2**2)
    b1               = -b1 / d
    b2               = -b2 / d

    # find the new points
    x2[:-1] = x[:-1] + length * b1
    y2[:-1] = y[:-1] + length * b2
    x2[-1]  = x2[0]
    y2[-1]  = y2[0]

    return x2, y2



def get_contours(x, y):
    """
    Creates a new contour from the given x and y coordinates.

    Args:
        x, y: numpy arrays representing the x and y coordinates of the points.

    Returns:
        contour: A 3D numpy array representing the new contour. The shape of the array is (n, 1, 2), 
                 where n is the number of points. This can be used in drawContours function of OpenCV.
    """
    
    # make new contour & add an extra dimension to match the original shape of max_contour
    contour = np.stack((x, y), axis=-1)
    contour = contour[:, np.newaxis, :]
    
    return contour



def roundx(y, x):
    """
    Rounds elements of vector "y" to closest element found in vector "x".

    Parameters:
    y (np.array): vector that you wish to round. Can be scalar, vector, or 2D array
    x (np.array): vector containing values you wish to round to. Must be scalar or vector.

    Returns:
    Y (np.array): output, same size as y, and Y[i] is "close to" y[i].
    idx (np.array): indices that we use to round: Y = x[idx]. Same size as "y".
    """

    # Ensure x is a 1D array
    if x.ndim > 1:
        raise ValueError('Input "x" must be a vector or scalar')

    # Reshape y and x to be column and row vectors respectively
    y = y.reshape(-1, 1)
    x = x.reshape(1, -1)

    # Compute absolute difference between y and x
    D = np.abs(y - x)

    # Find the indices of the minimum values in D along axis 1
    idx = np.argmin(D, axis=1)

    # Use the indices to get the corresponding values from x
    Y = x[0, idx.flatten()]

    # Reshape Y and idx to match the original shape of y
    # Y = Y.reshape(y.shape)
    # idx = idx.reshape(y.shape)

    return Y, idx