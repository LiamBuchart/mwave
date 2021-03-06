"""
Function which is used to generate a simplified (ideaized) sounding based on a set file
a desired input shape for the temparature
Examples: neutral->stable
          stable
          exponential
          low-inversion
          high-inversion

Usage: Liam comprehensive exam topic question (Stull)
To get a wind profile, this script calls another which creates
one based on sounding data (see: _____.py)

lbuchart@eoas.ubc.ca
"""

import numpy as np
import pandas as pd

from IPython.display import display
#from context import utils_dir

author="lbuchart"
pd.get_option('display.max_rows', None)

input_file = "default_input_sounding"

with open(input_file, "r") as fp:
    file_len = len(fp.readlines())  # get length of the file

lines = -1
sound_array = [[] for i in range(file_len)]  # empty array that is same length as file
strn = ""
check = " "

with open(input_file, "r") as f:
    for line in f:
        lines = lines + 1
        variables = -1
        inds = 0
        # grab the values from each line by looping though it and checking for adjacent spaces
        for ii in line:
            llen = len(line)
            inds = inds + 1
            if str(ii) == check or str(ii) == ".":
                pass
            else:
                strn = strn + str(ii)
           
            # now if we find a decimal we then save the number to array
            if str(ii) == ".":
                digs = [int(s) for s in strn.split() if s.isdigit()]
                # our fill string contains info and now the strn ii is a space
                variables = variables + 1
                sound_array[lines].append( digs ) 
                strn = ""

#print(sound_array)

wrf_sounding = pd.DataFrame(sound_array, columns=["height", "potential_temperature", "mixing_ratio", "U", "V"])





