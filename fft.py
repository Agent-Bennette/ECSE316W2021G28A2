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
##
##================================================================================================
##================================================================================================



#================================================
#
#   IMPORTS
#
#       matplotlib - show plots and images
#       numpy - useful for math stuff in general
#       PIL - need this for reading images
#       sys - need this for terminal arguments
#
#================================================

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys


#================================================
#
#   INPUT PARSING METHODS
#
#       Methods for getting program parametres
#       from either the user or from the
#       filesystem for running the program.
#
#       Here, you find:
#           - Arguments Parser
#           - Image Parser
#
#================================================

# Based on the arguments provided when this
# program is called via python in a terminal,
# returns either the default or explicitly
# provided program parametres.
# Returns None, None if invalid input is detected
# Returns an integer and string for program
# mode of operation and filename repsectively
# otherwise.
def get_mode_and_filename_from_terminal_params():

    #DEFAULT PARAMETRES
    prog_mode = 1
    file_name = "moonlanding.png"

    #PARAMETRE PARSING
    #sys.argv returns a list of all argument. arg 0 is always the name of the program.
    args_valid = True
    most_recent_arg = ""
    for i in range(1, len(sys.argv) ):
        #if the previous argument is option -m for mode
        if most_recent_arg == "-m" :
            #Break if mode already changed
            if prog_mode != 1:
                print("Duplicate -m argument detected. Aborting...")
                return None, None
            prog_mode = int(sys.argv[i])
            args_valid = True
            #Break if mode out of bounds
            if prog_mode < 1 or prog_mode > 4:
                print("Invalid -m argument detected: \'", prog_mode, "\'. Aborting...")
                return None, None
        #if the previous argument is option -i for file
        elif most_recent_arg == "-i" :
            file_name = sys.argv[i]
            args_valid = True
        #keep track of the previous argument
        most_recent_arg = sys.argv[i]
        #if -m or -r is the last arg, it's invalid.
        if most_recent_arg == "-m" or most_recent_arg == "-i" :
            args_valid = False

    #Return None if problems detected
    if args_valid == False:
        print("Detected option without argument. Aborting...")
        return None, None

    #Return args if arguments accepted
    print("No confusing arguments found. Proceeding...")
    return prog_mode, file_name

# Get initial command line args and deal with
# retrieving image byte data here as needed
def get_image_from_filename( filename, preview_input = False ):
    #Open up the image.
    image = None
    try:
        image = Image.open(filename)
    except FileNotFoundError:
        print("File ", filename, " was not found. Aborting...")
        return None
    #Represent image as array
    data = np.asarray(image)
    #Describe the image provided
    print("Data is of shape ", data.shape)
    print("Data preview: \n", data)
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

# Returns the data in (x, y) of two-dimensional
# array data.
def get_in_image_of_xy( data, x, y ):
    return data[y][x]

# Sets the data in (x, y) of two dimensional 
# array data with provided value value.
def set_in_image_of_xy( data, x, y, value ):
    data[y][x] = value
    return data


def lin_interpolate( val_1, val_2, pos_1, pos_2, pos ):
    normalized_pos = (pos_1-pos)/(pos_2-pos_1)
    val = val_1*normalized_pos + val_2*(1-normalized_pos)
    return val
    



#================================================
#
#   MAIN
#
#================================================

def main():
    # Obtain user-defined program parametres
    mode, filename = get_mode_and_filename_from_terminal_params()
    # Stop for user errors
    if mode == None or filename == None:
        return
    # Attempt to retrieve image data
    imgdata = get_image_from_filename(filename)
    # Stop for io errors
    if imgdata == None:
        return
    
    # Switch between program modes:
    if mode == 1:
        # Perform FFT and output 1x2 plot of the
        # original image and the fourier signal.



        return
    elif mode == 2:
        # Denoise Output: Output a 1x2 plot of
        # original image and denoised image.


        return
    elif mode == 3:
        # Double Functionality: (A) Compress the
        # to various degrees and output the
        # results in a 2x3 plot; and
        # (B) Save the Fourier Transform Matrix
        # to CSV.


        return
    elif mode == 4:
        # 


        return
    else:
        print("Illegal mode detected. Exiting.")
        return

if __name__ == "__main__":
    main()
