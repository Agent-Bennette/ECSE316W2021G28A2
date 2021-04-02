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
#       io - need this for reading files
#       sys - need this for terminal arguments
#
#================================================

import matplotlib.pyplot as plt
import numpy as np
import io
import sys



#================================================
#
#   Program Runtime Parametres Parsing
#
#       Get initial command line args and deal
#       with retrieving image byte data here
#       as needed.
#
#================================================
def get_mode_and_file_from_terminal_params():

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
    
