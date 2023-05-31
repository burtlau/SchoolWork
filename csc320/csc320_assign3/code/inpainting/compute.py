# CSC320 Fall 2022
# Assignment 3
# (c) Kyros Kutulakos
#
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
from . import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
from . import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    
    # Replace this dummy value with your own code
    valid_patch = copyutils.getWindow(confidenceImage, (psiHatP.row(), psiHatP.col()), psiHatP.radius())[0]
    width = (psiHatP.radius() * 2) + 1
    #print(width)
    C = np.sum(valid_patch / width ** 2)
    #print(C)
    #########################################
    
    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#
    
def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    
    # Replace these dummy values with your own code
    Dy = 1
    Dx = 0    
    #########################################
    grey_patch = cv.cvtColor(psiHatP.pixels(), cv.COLOR_BGR2GRAY)
    center = [psiHatP.row(), psiHatP.col()]
    filled_patch = copyutils.getWindow(filledImage, center, psiHatP.radius())[0]
    grey_patch = grey_patch * (filled_patch / 255)
    # print("grey_patch", grey_patch)
    sobelx = cv.Sobel(grey_patch, cv.CV_64F, 1, 0, ksize=5)
    # print("sobelx", sobelx)
    sobely = cv.Sobel(grey_patch, cv.CV_64F, 0, 1, ksize=5)
    # print("sobely", sobely)
    mag_grad = np.sqrt(sobelx**2 + sobely**2)
    i, j = np.unravel_index(np.argmax(mag_grad), mag_grad.shape)
    Dx = sobelx[i, j]
    Dy = sobely[i, j]

    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    # Filled the last pixel
    if (np.count_nonzero(psiHatP.filled()) <= 1):
        Nx, Ny = None, None
        return Nx, Ny
    else:
        grey_patch = cv.cvtColor(psiHatP.pixels(), cv.COLOR_BGR2GRAY)
        center = [psiHatP.row(), psiHatP.col()]
        filled_patch = copyutils.getWindow(filledImage, center, psiHatP.radius())[0]
        grey_patch = grey_patch * (filled_patch / 255)
        sobelx = cv.Sobel(grey_patch, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(grey_patch, cv.CV_64F, 0, 1, ksize=5)
        center_new = (psiHatP.radius(), psiHatP.radius())
        y = sobelx[center_new]
        x = sobely[center_new]
        norm = np.sqrt(y**2 + x**2)

    # Replace these dummy values with your own code
        Ny = -sobelx[center_new]/norm
        Nx = sobely[center_new]/norm
    #########################################

    return Ny, Nx
