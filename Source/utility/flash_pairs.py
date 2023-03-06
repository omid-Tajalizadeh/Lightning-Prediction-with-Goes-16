from __future__ import (print_function, division, unicode_literals)
from satpy import Scene


from netCDF4 import Dataset as ncds
import netCDF4 as nc

import numpy as np




def scene_reader(file_path):

    scene = Scene(reader="abi_l1b", filenames = [file_path]) # --- needs to be list of files ---

    scene.load(['C02'])

    area = scene['C02'].attrs['area']

    return area

area = scene_reader("/data/s2896370/MobileNetV2/3ch/1/OR_ABI-L1b-RadF-M6C02_G16_s20191081010223_e20191081019531_c20191081019570.nc")

def to_pixel(area, flash):

    col_idx, row_idx = area.get_xy_from_lonlat( flash[1], flash[0])
    pixels = [row_idx, col_idx]

    return pixels

def create_flash_pairs(glmfile):

    glmds = ncds(glmfile)
    ls_flon = np.array(glmds.variables["flash_lon"])
    ls_flat = np.array(glmds.variables["flash_lat"])

    flash_pairs = []
    for i in range(len(ls_flat)):
        flash_pairs.append([ls_flat[i], ls_flon[i]])

    return flash_pairs


def pixel_of_flashes(flash_pairs):
    pixels = []
    for pair in flash_pairs:
        if ((4.9<pair[0]<20) and (-80.1<pair[1]<-48.5)):
            pixels.append(to_pixel(area,pair))
    pixels =  np.array(pixels)

    return pixels

def get_pof(glmf):
    flash_pairs = create_flash_pairs(glmf)
    pof = pixel_of_flashes(flash_pairs)

    return pof




glmf = "/data/s2896370/MobileNetV2/3ch/1/GLM-L2-LCFA.noaa-goes16.20191081010.CaribbeanLarge.nc"


print(get_pof(glmf).shape)


