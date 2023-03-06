from netCDF4 import Dataset as ncds
import numpy as np
import copy
import cv2

import random 






def convert_to_normBT(ds):
    RAD = ds["Rad"][:]
    pfk1 = ds.variables['planck_fk1'][:]
    pfk2 = ds.variables['planck_fk2'][:]
    pbc1 = ds.variables['planck_bc1'][:]
    pbc2 = ds.variables['planck_bc2'][:]
    BT103 = copy.deepcopy(RAD)
    index = np.where(RAD != RAD.fill_value)
    BT103[index] = (pfk2/(np.log((pfk1/RAD[index]) + 1.0) - pbc1))/pbc2
    BT103 = np.array(BT103)
    normalized_BT = (BT103 - 100)/( 350 - 100)
    return np.array(normalized_BT)

def ch2_normalized(ds):
    
    kappa0 = float(ds.variables["kappa0"][:])
    norm = ds.variables["Rad"][:]*kappa0
    norm = np.maximum(norm, 0.0)
    norm = np.minimum(norm, 1.0)
    return norm


def conv_to_BT(ch2_file, ch13_file, ch15_file):
    ch2 =  ncds(ch2_file)
    ch13 = ncds(ch13_file)
    ch15 = ncds(ch15_file)

    day =  ch2_file.split("_")[3][5:8]
    year =  ch2_file.split("_")[3][1:5]
    hour = ch2_file.split("_")[3][8:10]


    ch2_normalizd = ch2_normalized(ch2)
    ch13_normalized = convert_to_normBT(ch13)
    ch15_normalized = convert_to_normBT(ch15)

    ch13_resized = cv2.resize(ch13_normalized, (6201,3151), interpolation = cv2.INTER_AREA)
    ch15_resized = cv2.resize(ch15_normalized, (6201,3151), interpolation = cv2.INTER_AREA)

    return ch2_normalizd, ch13_resized, ch15_resized
