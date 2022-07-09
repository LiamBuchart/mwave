"""
Function which takes a terrain type input:
***** gaussian, sinusoidal, plateau, angular *****
and creates idealized terrain heights based on namelist input
information for your idealized WRF run,

Used to create topography for downslope wind storm experiments.
Note that all configurations with have a max height of 1000m.

lbuchart@eoas.ubc.ca
"""
import numpy as np
from context import name_dir

author="lbuchart"


terrain_type = "gaussian"  # what type of simple terrain (pick from list above in ***** ******
experiment = "test"  # in the exp directory pick which experiment you will be making terrain for
outfile_name = "input_ht"

npath = name_dir + experiment + "/"

namelist = open(npath + "namelist.input", "r")  # read in the namelist
print("Read Namelist")
lines = namelist.readlines()
print(np.shape(lines)) 
text = ["e_we", "e_sn"]  # namelist variables for the domain size

we = []
idw = 0
sn = []
ids = 0
# loop through each line in the namelsit
for line in lines:
    #print(line)
    # check for the ew direction name
    if text[0] in line:
        print("Yes")
        we = line
        idw += 1
    
    # check for the ns direction name
    if text[1] in line:
        print("Yes")
        sn = line
        ids += 1
    
namelist.close()

# extract the digit number of grid cells for both the ns and ew direction
we_n = [i for i in we if i.isdigit()]
sn_n = [i for i in sn if i.isdigit()]

we_n = int(''.join(we_n))
sn_n = int(''.join(sn_n))

# create a grid of zeros based on namelist size
hgt = np.empty((we_n, sn_n), np.float32)
grid_size = np.shape(hgt)
print(hgt.ndim, grid_size[0], grid_size[1])

# loop through the grid and add topography heights based on the chosen methods
# note that topography is added in the east-west direction
for ii in range(grid_size[1]):

    if terrain_type.__eq__("gaussian"):
        
        def gaussian(mht, x, mu, sig):
            # x = rnage over which gaussian hill is made
            # mu = offset of the guassian on domain
            # sig = steepness factors (>1 steeper)
            return  mht*np.exp( -np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
            
        mu = grid_size[0] / 4  # make the gaussian hill ~half the size of the domain
        sig = 4
        mht = 600  # max height of the hill
        x = np.arange(grid_size[0])
        
        hgt[:, ii] = gaussian(mht, x, mu, sig)  
    
    #elif terrain_type.__eq__("sinusoidal"):

# now add the required info and array to the outfile_name text file which we will create
dims = str(we_n) + " " + str(sn_n)
print(dims)

np.savetxt(outfile_name, hgt, header=dims, comments="", fmt="%i")
print("Saved the height array")