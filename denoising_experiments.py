##================================================================================================
##================================================================================================
##
##
##      ECSE 316 - Signald and Networks
##      Winter 2021 Edition
##      Assignment 2
##
##      Group 28
##      Edwin Pan and Ji Ming Li
##      Due 2021 April 5th
##
##          THIS IS AN EXTRA FILE USED FOR EXPERIMENTING WITH DIFFERENT DENOISING METHODS.
##          THE MAIN FILE FOR GRADING IS STILL fft.py.
##
##
##================================================================================================
##================================================================================================


import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
import numpy as np
import math
import heapq
import time


#================================================
#
#   PROGRAM CONSTANTS <deprecated>
#
#       Any pre-runtime-set variables are kept
#       here. Useful, in particular, for things
#       which ened tuning.
#
#       Coefficients defining the noise filter
#       are defined here.
#
#================================================

# In terms of normalized distance from centre to
# wall of the image, where is 0.5 denoted.
DENOISING_CENTRE = 0.5
# How rapidly the sigmoid function use applies.
# Use high value for binary threshold; use
# low value for soft transition.
DENOISING_HARSHNESS = 3.0




# Get initial command line args and deal with
# retrieving image byte data here as needed
def get_image_from_filename( filename, preview_input = False ):
    #Open up the image.
    image = None
    try:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print("File ", filename, " was not found. Aborting...")
        return None
    #Represent image as array
    data = np.asarray(image)
    #Describe the image provided
    #print("Data is of shape ", data.shape)
    #print("Data preview: \n", data)
    #image2 = Image.fromarray(data)
    #If preview enabled, have a pop-out window to show the input.
    if preview_input:
        plt.imshow( data, cmap='Greys')
        plt.show()
    #Return the image in array form
    return image

#================================================
#
#   IMAGE-RELATED HELPER METHODS
#
#       A set of methods useful for interacting
#       with the numpy-array representation of
#       the image is provided here.
#
#================================================

# Upsizes the image to nearest power of two
# width and height as needed for fft's.
def upsize_to_square_power_of_two(img):
    max_dim = max( len(img), len(img[0]) )
    new_dim = 2
    while True:
        if new_dim is max_dim:
            break
        elif new_dim > max_dim:
            break
        else:
            new_dim *= 2
    return cv2.resize(img, (new_dim,new_dim) )

# Returns the minimum value in a two dimensional
# array input arr.
def min_in_2d( arr ):
    min_val = float('inf')
    for subarr in arr:
        for element in subarr:
            if element < min_val:
                min_val = element
    return min_val

# Returns the maximum value in a two dimensional
# array input arr.
def max_in_2d( arr ):
    max_val = float('-inf')
    for subarr in arr:
        for element in subarr:
            if element > max_val:
                max_val = element
    return max_val

# Returns the complex-numbered 2d array into a
# real numbered 2d array comprising of the
# magnitudes at each original element.
def mag_2d_of_complex_2d( arr ):
    height = len(arr)
    width = len(arr[0])
    new_arr = np.empty( (height, width) )
    for i in range(height):
        for j in range(width):
            new_arr[j][i] = ( arr[j][i].real**2 + arr[j][i].imag**2 )**0.5
    return new_arr

# Returns the real-valued 2d array of arr by
# ignoring all the imaginary components.
def ignore_imaginaries( arr ):
    # Remove Complex Addend (bugfix)
    height = len(arr)
    width = len(arr[0])
    real_arr = np.empty( arr.shape )
    for i in range(height):
        for j in range(width):
            real_arr[i][j] = arr[i][j].real
    return real_arr



#================================================
#
#   MATH IMPLEMENTATIONS
#       includes FFT and iFFT's as well as
#       useful saving and plotting functions.
#       
#       Methods completing DFT, FFT, iFFT,
#       2D FFT, 2D iFFT are here.
#
#       Methods creating plots and saving
#       transform arrays are also here.
#
#       Section's function implementations are
#       all written by Ji Ming Li.
#
#================================================

# Direct Fourier Transform on input 1D array x.
def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, x)
    return X

# Fast Fourier Transform on input 1D array x.
def fft(x):
    N = len(x)

    if N ==1:
        return x
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        f = np.exp(-2j*np.pi*np.arange(N)/ N)
        X = np.concatenate([X_even+f[:int(N/2)]*X_odd, X_even+f[int(N/2):]*X_odd])

    return X

# Inverse Fourier Transform on input 1D array x.
def ifft(x):
    N = len(x)

    if N == 1:
        return x
    else:
        X_even = ifft(x[::2])
        X_odd = ifft(x[1::2])
        f = np.exp(2j*np.pi*np.arange(N)/ N)
        X = np.concatenate([X_even+f[:int(N/2)]*X_odd, X_even+f[int(N/2):]*X_odd])
        X = X/2
        return X

# Two-Dimensional Fourier Transform on 2d array x naive algorithm.
def dft2d(x):
    N = len(x)
    M = len(x[0])

    X = np.empty([N, M], dtype=complex)

    for i in range(M):
        X[:,i] = dft(x[:,i])
    for i in range(N):
        X[i] = dft(X[i])

    return X

# Two-Dimensional Fourier Transform on 2d array x.
def fft2d(x):
    N = len(x)
    M = len(x[0])

    X = np.empty([N, M], dtype=complex)

    for i in range(M):
        X[:,i] = fft(x[:,i])
    for i in range(N):
        X[i] = fft(X[i])

    return X

# Two-Dimensional Inverse Fourier Transform on
# 2d array x.
def ifft2d(x):
    N = len(x)
    M = len(x[0])

    X = np.empty([N, M], dtype=complex)

    for i in range(M):
        X[:,i] = ifft(x[:,i])
    for i in range(N):
        X[i] = ifft(X[i])

    return X

# Simple uniformly scaled plot of 2d array.
def plot(x):
#    N = len(x)
#    M = len(x[0])
    plt.imshow(abs(x))
    plt.show()

# Displays the provided image in input x.
def plotImage(x):
    img = Image.fromarray(x, 'L')
#    imgplot = plt.imshow(img)
    img.show()

# Creates a logarithmically y-scaled plot
# on input two dimensional array x.
# Modified from how it was originally written in colab
# to fix some errors.
#def logPlot(x):
#    cmap = colors['plasma']
#    cmap.set_bad(cmap(0))
#    h, xedges, yedges = np.histogram2d(x)
#    pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap,
#                         norm=LogNorm(vmax=1.5e2), rasterized=True)
#    fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
#    axes[1].set_title("2d histogram and log color scale")

# Saves the transform matrix x to file
# with name name.
#def saveToCSV(x, name):
#    np.savetxt(name+".csv", a, delimiter=",")



#================================================
#
#   MASK FUNCTIONS
#
#       Functions which map a 0.0 to 1.0
#       multiplicand to each fourier transform.
#       Used and experimented with for denoising.
#
#================================================

# Sigmoid function.
def sigmoid(x):
    return 1/( 1+math.exp(-x) )

# Radial Mask Function: Used to reduce noise
# the farther away out from the centre of a
# square frame something is.
# Distance is a normalized 0-1 where 0
# indicates being at the centre of the image
# while 1 indicates being in a centre of an
# edge of a square image.
def radial_inside_mask(x):
    return -sigmoid( DENOISING_HARSHNESS*(x-DENOISING_CENTRE) )

# Quadratic mask which uses a quadratic
# function with a trough (no peak) where
# positions where the function dips below
# 0 are set to 0 and locations
def mask_quadratic(x_n,y_n,sinkage,harshness,radius):
    return max(0, 
                min(1,
                    min(harshness*(x_n-0.5-radius)*(x_n-0.5+radius),
                        harshness*(y_n-0.5-radius)*(y_n-0.5+radius))
                    -sinkage 
                    )
                )

# Given position i and a nx2 array
# representing a bijection of point pairs,
# returns the interpolated second value
# between two point pairs of g_i
# based on g_i[x][0] values.
def lin_interpolate( i, g_i ):
    if i <= g_i[0][0]:
        return g_i[0][1]
    elif i >= g_i[len(g_i)-1][0]:
        return g_i[len(g_i)-1][1]
    else:
        left = 0
        right = 1
        for i in range(1,len(g_i)):
            if g_i[i-1][0] < i and i <= g_i[i][0]:
                left = i-1
                right = i
                break
        left_i = g_i[left][0]
        right_i = g_i[right][0]
        left_v = g_i[left][1]
        right_v = g_i[right][1]
        norm_i = -1
        if right_i is left_i:
            norm_i = right_v
        else:
            norm_i = (i-left_i)/(right_i-left_i)
        return (1-norm_i)*left_v + norm_i*right_v

# Create a mask based on input points
# to interpolate g_i, a Nx2 array for
# any arbitrary natrual N.
# Values in the second column of g_i
# are all assumed to be normalized.
# Values x_n and y_n must be normalized
# between 0 and 1 before inputting.
# Applies gradient g_i based on y value.
def mask_grad_y( x_n, y_n, g_i ):
    return lin_interpolate(y_n,g_i)

# Create a mask based on input points
# to interpolate g_i, a Nx2 array for
# any arbitrary natrual N.
# Values in the second column of g_i
# are all assumed to be normalized.
# Values x_n and y_n must be normalized
# between 0 and 1 before inputting.
# Applies gradient g_i based on x value.
def mask_grad_x( x_n, y_n, g_i ):
    return lin_interpolate( x_n, g_i )

# Creates a mask comprising mask_grad_y
# and mask_grad_x as multiplier layers.
# "Gradients" for x axis and y axis are
# explicit and must each be provided.
def mask_grad_prod( x_n, y_n, g_x, g_y ):
    x = lin_interpolate(x_n,g_x)
    y = lin_interpolate(y_n,g_y)
    v = max(0,min(1,x*y))
    return v

# Creates a mask comprising mask_grad_y
# and mask_grad_x as added layers.
# "Gradients" for x axis and y axis are
# explicit and must each be provided.
def mask_grad_sum( x_n, y_n, g_x, g_y ):
    x = lin_interpolate(x_n,g_x)
    y = lin_interpolate(y_n,g_y)
    v = max(0,min(1,x+y))
    return v



#================================================
#
#   MAIN
#
#================================================

def main():
    # Attempt to retrieve image data
    img = get_image_from_filename("moonlanding.png")
    # Stop for io errors
    if img is None:
        return

    # Keep record of the original image
    # resolution.
    height = len(img)
    width = len(img[0])
    # Upscale image for fft and apply fft
    up_img = upsize_to_square_power_of_two( img )
    up_dim = len(up_img)
    # Fourier Transform
    up_fft = fft2d(up_img)

    ## DENOISE EXPERIMENTATION SECTION
    ## ----------------------------------------------------
    ## Use Various Denoising Configurations and Graph them.
    ## Divided into three sections:
    ##  i.      Prepare matplotlib figure
    ##  ii.     Experiment with various denoising measures:
    ##              iia:    Threshold pass as needed
    ##              iib:    Mask pass as needed
    ##              iic:    Inverse FFT back to Image
    ##              iid:    Plug image into figures
    ##  iii.    Show Plots
    ## ----------------------------------------------------
    ##  Tests comprise:
    ##      4 Vertical Gradient Mask Experiments
    ##      4 Horizontal Gradient Mask Experiments
    ##      4 Dual-Axis Gradient Product Mask Experiments
    ##      4 Dual-Axis Gradient Added Mask Experiments
    ##      4 Quadratic Mask Experiments
    ##      4 Sigmoid Mask Experiments
    ##      4 Threshold Mask Experiments
    ##      4 Hand-Selected Experiments
    ## ----------------------------------------------------

    #i.     MatPlotLib Figure Constructing
    fig, subplots = plt.subplots(4,8)
    fig.suptitle("Results of Various Denoising Configurations")

    #ii.1)  Horizontal Gradient Masks
    gradients = [
        [
            [0.0,1.0],
            [0.5,0.0],
            [0.5,0.0],
            [1.0,1.0]
        ],
        [
            [0.0,1.0],
            [0.25,0.0],
            [0.75,0.0],
            [1.0,1.0]
        ],
        [
            [0.25,1.0],
            [0.25,0.0],
            [0.75,0.0],
            [0.75,1.0]
        ],
        [
            [0.25,0.0],
            [0.25,1.0],
            [0.75,1.0],
            [0.75,0.0]
        ],
    ]
    for trial in range(4):
        denoised_2dfft = np.copy(up_fft)
        for i in range(up_dim):
            for j in range(up_dim):
                x = j/up_dim
                y = i/up_dim
                mask = mask_grad_x(x,y,gradients[trial])
                denoised_2dfft[j][i] = denoised_2dfft[j][i]*mask
        denoised_image = cv2.resize( ignore_imaginaries( ifft2d( denoised_2dfft ) ), (width,height))
        subplots[trial][0].imshow( denoised_image, cmap='gray' )
        denoised_image_name = "Horizontal Gradient Masking Trial " + str(trial)
        cv2.imwrite( "Experimental - " + denoised_image_name+".png", denoised_image )
        subplots[trial][0].set_title( denoised_image_name )

    #ii.2)  Vertical Gradient Masks
    for trial in range(4):
        denoised_2dfft = np.copy(up_fft)
        for i in range(up_dim):
            for j in range(up_dim):
                x = j/up_dim
                y = i/up_dim
                mask = mask_grad_y(x,y,gradients[trial])
                denoised_2dfft[j][i] = denoised_2dfft[j][i]*mask
        denoised_image = cv2.resize( ignore_imaginaries( ifft2d( denoised_2dfft ) ) , (width,height))
        subplots[trial][1].imshow( denoised_image, cmap='gray' )
        denoised_image_name = "Vertical Gradient Masking Trial " + str(trial)
        cv2.imwrite( "Experimental - " + denoised_image_name+".png", denoised_image )
        subplots[trial][1].set_title( denoised_image_name )

    #ii.3)  Dual-Axis Gradient Product Masks
    for trial in range(4):
        denoised_2dfft = np.copy(up_fft)
        for i in range(up_dim):
            for j in range(up_dim):
                x = j/up_dim
                y = i/up_dim
                mask = mask_grad_prod(x,y,gradients[trial],gradients[trial])
                denoised_2dfft[j][i] = denoised_2dfft[j][i]*mask
        denoised_image = cv2.resize( ignore_imaginaries( ifft2d( denoised_2dfft ) ) , (width,height))
        subplots[trial][2].imshow( denoised_image, cmap='gray' )
        denoised_image_name = "Dual Gradient Product Masking Trial " + str(trial)
        cv2.imwrite( "Experimental - " + denoised_image_name+".png", denoised_image )
        subplots[trial][2].set_title( denoised_image_name )

    #ii.4)  Dual-Axis Gradient Sum Masks
    for trial in range(4):
        denoised_2dfft = np.copy(up_fft)
        for i in range(up_dim):
            for j in range(up_dim):
                x = j/up_dim
                y = i/up_dim
                mask = mask_grad_sum(x,y,gradients[trial],gradients[trial])
                denoised_2dfft[j][i] = denoised_2dfft[j][i]*mask
        denoised_image = cv2.resize( ignore_imaginaries( ifft2d( denoised_2dfft ) ) , (width,height))
        subplots[trial][3].imshow( denoised_image, cmap='gray' )
        denoised_image_name = "Dual Gradient Sum Masking Trial " + str(trial)
        cv2.imwrite( "Experimental - " + denoised_image_name+".png", denoised_image )
        subplots[trial][3].set_title( denoised_image_name )

    #ii.5)  Quadratic-Distance Masks
    sinkage = [0.4, 0.0, 0.4, 0.4]
    harshness = [12, 12, 0.01, 12]
    radius = [0.25, 0.25, 0.25, 0.01]
    for trial in range(4):
        denoised_2dfft = np.copy(up_fft)
        for i in range(up_dim):
            for j in range(up_dim):
                x = j/up_dim
                y = i/up_dim
                mask = mask_quadratic(x,y, sinkage[trial], harshness[trial], radius[trial])
                denoised_2dfft[j][i] = denoised_2dfft[j][i]*mask
        denoised_image = cv2.resize( ignore_imaginaries( ifft2d( denoised_2dfft ) ) , (width,height))
        subplots[trial][4].imshow( denoised_image, cmap='gray' )
        denoised_image_name = ("Quadratic Masking with " + str(sinkage[trial]) + "sink, " + 
            str(harshness[trial]) + "harsh, and " + str(radius[trial]) + "radius."
        )
        cv2.imwrite( "Experimental - " + denoised_image_name+".png", denoised_image )
        subplots[trial][4].set_title( denoised_image_name )

    #ii.6)  Sigmoid Masks
    sigmoid_on_x_and_not_y = [True, True, False, False]
    sigmoid_squeeze = [1, -1, 1, -1]
    for trial in range(4):
        denoised_2dfft = np.copy(up_fft)
        for i in range(up_dim):
            for j in range(up_dim):
                x = 1
                if sigmoid_on_x_and_not_y :
                    x = sigmoid( sigmoid_squeeze[trial]*( j/up_dim -0.5 ) )
                y = 1
                if not sigmoid_on_x_and_not_y :
                    x = sigmoid( sigmoid_squeeze[trial]*( i/up_dim -0.5 ) )
                denoised_2dfft[j][i] = x*y*denoised_2dfft[j][i]
        denoised_image = cv2.resize( ignore_imaginaries( ifft2d( denoised_2dfft ) ) , (width,height))
        subplots[trial][5].imshow( denoised_image, cmap='gray' )
        direction = "Vertical"
        if sigmoid_on_x_and_not_y[trial]:
            direction = "Horizontal"
        flip = "Flipped"
        if sigmoid_squeeze[trial] > 0:
            flip = "Not Flipped"
        denoised_image_name = direction + flip + "Sigmoid Masking"
        cv2.imwrite( "Experimental - " + denoised_image_name+".png", denoised_image )
        subplots[trial][5].set_title( denoised_image_name )

    #ii.7)  Threshold Masks
    threshold_is_min_dir = [True, True, False, False]
    threshold_values = [1e-1,1e1,1e-1,1e1]
    for trial in range(4):
        denoised_2dfft = np.copy(up_fft)
        for i in range(up_dim):
            for j in range(up_dim):
                if threshold_is_min_dir[trial] is True and denoised_2dfft[j][i] < threshold_values[trial]:
                    denoised_2dfft[j][i] = 0
                elif threshold_is_min_dir[trial] is False and denoised_2dfft[j][i] > threshold_values[trial]:
                    denoised_2dfft[j][i] = 0
        denoised_image = cv2.resize( ignore_imaginaries( ifft2d( denoised_2dfft ) ) , (width,height))
        subplots[trial][6].imshow( denoised_image, cmap='gray' )
        threshold_type = "Keep-Below "
        if threshold_is_min_dir :
            threshold_type = "Keep-Above "
        denoised_image_name = ( 
            threshold_type + str(threshold_values) + "Threshold Mask"
        )
        cv2.imwrite( denoised_image_name+".png", denoised_image )
        subplots[trial][6].set_title( denoised_image_name )

    #ii.8)  Custom Masks
    for trial in range(3):
        denoised_2dfft = np.copy(up_fft)
        for i in range(up_dim):
            for j in range(up_dim):
                if denoised_2dfft[j][i] < 0.01:
                    denoised_2dfft[j][i] = 0
                else:
                    mask1 = mask_grad_prod(j/up_dim,i/up_dim,gradients[3],gradients[3])
                    mask2 = mask_quadratic(j/up_dim,i/up_dim,0.4,12,0.5)
                    denoised_2dfft[j][i] = mask1*mask2*denoised_2dfft
                    if denoised_2dfft[j][i] < 0.01:
                        denoised_2dfft[j][i] = 0
        denoised_image = cv2.resize( ignore_imaginaries( ifft2d( denoised_2dfft ) ) , (width,height))
        subplots[trial][7].imshow( denoised_image, cmap='gray' )
        denoised_image_name = "Custom" + str(trial)
        cv2.imwrite( "Experimental - " + denoised_image_name+".png", denoised_image )
        subplots[trial][7].set_title( denoised_image_name )
    subplots[3][7].set_title("Original Image")
    subplots[3][7].imshow( img )


    #iii    Show Images
    plt.show()



if __name__ == "__main__":
    main()
