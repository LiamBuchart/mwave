"""
Function which utilizes two other functions GB_Regression and get_heights
to then build a dataframe to then save it as a WRF input_sounding file

Usage: creating a WRF-like sounding based on the default ideal 
input_sounding file. Wind (U) and q come from Vernon sounding data
Theta profile comes from a prescribed shape 
For Roland's comps question for Liam

Theta options: ***** (input in line 36) ***** write as shown in theta_type
               1. neutstab (neutral to stable) 
               2. stable
               3. log
               4. low-step
               5. high-step

Output: input_sounding [text file] that is same format for WRF

July 27, 2022, 
lbuchart@eoas.ubc.ca
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from GB_Regression import y_mixing, y_speed, hh
from get_heights import heights

from context import json_dir

###########

##########
theta_type = "neutstab"
##########

def extract_val(arr):
    new_var = ""
    for chars in range(0, len(arr)):
        vals = arr[chars]
        for item in vals:
            if item.isdigit():
                new_var = new_var + str(item)

    return new_var

##########

# get mountian height from our json dictionary
with open(str(json_dir) + "config.json") as f:
    config = json.load(f)
m_height = config["topo"]["mht"]  # topography height
T0 = config["sounding"]["T_sfc"]  # surface temperature
P0 = config["sounding"]["P_sfc"]  # surface pressure

input_sounding = pd.DataFrame(columns=["height", "theta", "q", "U", "V"])

hh = hh.ravel()
ind = np.where(hh == 0)
qfind = y_mixing[ind]

qfind = str(round(qfind[0]))
q1 = extract_val(qfind)

# overwrite the base input_sounding dateframe to account for first row formatting
# note that the first row of height is actuall the surface pressure
weather_data = {"height":[P0], "theta":[T0], "q":[int(q1)], "U":[np.nan], "V":[np.nan]}
input_sounding = pd.DataFrame(weather_data, index=[0])

#new_row = pd.DataFrame([values], columns=["height", "theta", "q", "U", "V"])
# input_sounding.append(new_row, ignore_index=True)

# build potential temperature profile
theta = np.zeros((len(hh),), dtype=np.float32)
if theta_type == "neutstab":

    # neutreal lower layer then slowly increaing potential temp
    ind = np.where(hh == int(1.5*m_height))
    for ii in range(0, int(ind[0])):
        theta[ii] = T0
    
    slope = 0.005  # potential temp increase / m (theta = T0 + slope*(hh[ii]-hh[ind]))
    for ii in range((int(ind[0])), len(hh)):
        h_change = hh[int(ind[0])]
        theta[ii] = round( T0 + ( slope * (hh[ii] - h_change) ), 2 )

elif theta_type == "stable":

    # stable and slowing increasing potential temperature with height
    slope = 0.001

    for ii in range(0, len(hh)):
        theta[ii] = round( T0 + ( slope * hh[ii] ), 2 )

elif theta_type == "log":

    # a logarithmic potential temperature profile (theta = amp*log(str*hh[ii]))
    amp = 5
    str = 2
    for ii in range(0, len(hh)):
        theta[ii] = amp * np.log10(str * hh[ii])
        
elif theta_type == "low_step" or theta_type == "high_step":
    
    # sharp potential temp inversion below mountaintop height
    if theta_type == "low_step":
        inv_height = m_height * 0.75  # ensure this value is a multiple of 10 (in hh)
    elif theta_type == "high_step":
        inv_height = m_height * 1.25
    inv_step = 5  # size of the potential temp inversion
    
    ind = np.where(hh == inv_height)
    for ii in range(0, int(ind[0])):
        theta[ii] = T0
    for ii in range(int(ind[0]), len(hh)):
        theta[ii] = T0 + inv_step
    
print("Quick peak: ", theta)

plt.figure(figsize=(12, 12))
plt.plot(theta, hh)
plt.xlabel("Potential Temperature")
plt.ylabel("Height [m]")
plt.title("Idealized Vertical Potential Temperature Profile")
plt.savefig("Theta_Profile.png")

# update the dictionary iteratively to make vertical profile of wind (U) and q - V is zero
# grab theta from the prescribed potential temperature profile

print("Making the input sounding dictionary and file")
heights = heights[1:]  # recall that the 0 index in heights is actually the sfc pressure
y_speed = y_speed / 1.94384  # convery from kts to m/s
for ii in range(0, len(heights)):
    # loop through heights and grab the predicted values
    ind = np.where(hh == heights[ii])

    h_new = heights[ii]
    theta_new = theta[ind]
    
    u_find = y_speed[ind]
    u_new = extract_val( str(round(u_find[0])) )
    q_find = y_mixing[ind]
    q_new = extract_val( str(round(q_find[0])) )
    v_new = 0
    
    for key,vals in weather_data.items():
        if key == "height":
            vals.append(h_new)
        elif key == "theta":
            vals.append(int(theta_new))
        elif key == "q":
            vals.append(int(q_new))
        elif key == "U":
            vals.append(int(u_new))
        elif key == "V":
            vals.append(v_new)

input_sounding = pd.DataFrame(weather_data)

#print(input_sounding)

input_sounding.to_csv("input_sounding", 
                      sep=" ", 
                      header=False, 
                      index=False, 
                      index_label=None)