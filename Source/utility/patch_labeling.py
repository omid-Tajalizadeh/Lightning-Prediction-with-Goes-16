import numpy as np
import random
from patchify import patchify

from flash_pairs import *
from to_BT import conv_to_BT 


ch2_file = "/data/s2896370/MobileNetV2/3ch/2/OR_ABI-L1b-RadF-M6C02_G16_s20192631500194_e20192631509502_c20192631509546.nc"
ch13_file = "/data/s2896370/MobileNetV2/3ch/2/OR_ABI-L1b-RadF-M6C13_G16_s20192631500194_e20192631509513_c20192631510000.nc"
ch15_file = "/data/s2896370/MobileNetV2/3ch/2/OR_ABI-L1b-RadF-M6C15_G16_s20192631500194_e20192631509508_c20192631509594.nc"
glmf = "/data/s2896370/MobileNetV2/3ch/2/GLM-L2-LCFA.noaa-goes16.20192631500.CaribbeanLarge.nc"


ch2_normalized, ch13_resized, ch15_resized = conv_to_BT(ch2_file, ch13_file, ch15_file)

RGB = np.stack([ch2_normalized, ch13_resized, ch15_resized], axis = -1)
p_RGB = RGB[0:3500,0:3500, :]

patches =  patchify(p_RGB, (64,64,3), step=1)


p_labels = np.zeros((patches.shape[0],patches.shape[1]))
fp = get_pof(glmf)

print("labeling patches:-----------")
for i in range(p_labels.shape[1]-63):
    for j in range(p_labels.shape[0]-63):
        for flash in fp:
            if((j<flash[0]<j+63) and (i<flash[1]<i+63)):
                p_labels[j,i]=1
print("patch labeling done.")
np.save("/data/s2896370/MobileNetV2/Source/utility/patch_labels/patch_labels_simp", p_labels)
# pl = np.load("patch_labels.npy")



    




