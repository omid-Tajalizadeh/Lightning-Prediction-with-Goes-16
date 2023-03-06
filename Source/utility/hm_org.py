from to_BT import conv_to_BT 
from flash_pairs import *
import numpy as np
import matplotlib.pyplot as plt
from circle import *

ch2_file = "/data/s2896370/MobileNetV2/3ch/2/OR_ABI-L1b-RadF-M6C02_G16_s20192631500194_e20192631509502_c20192631509546.nc"
ch13_file = "/data/s2896370/MobileNetV2/3ch/2/OR_ABI-L1b-RadF-M6C13_G16_s20192631500194_e20192631509513_c20192631510000.nc"
ch15_file = "/data/s2896370/MobileNetV2/3ch/2/OR_ABI-L1b-RadF-M6C15_G16_s20192631500194_e20192631509508_c20192631509594.nc"


glmf = "/data/s2896370/MobileNetV2/3ch/2/GLM-L2-LCFA.noaa-goes16.20192631500.CaribbeanLarge.nc"



# hmap_path = "/data/s2896370/MobileNetV2/Source/np_heatmaps/hmap-res-35mins.npy"
# hmap_path = "/data/s2896370/MobileNetV2/Source/np_heatmaps/hmap1-res-35mins.npy"
# hmap_path = "/data/s2896370/MobileNetV2/Source/hmap-dense-35mins.npy"
hmap_path = "/data/s2896370/MobileNetV2/Source/hmap-simp-35mins.npy"

day =  ch2_file.split("_")[3][5:8]
year =  ch2_file.split("_")[3][1:5]
hour = ch2_file.split("_")[3][8:10]

ch2_bt, ch13_bt, ch15_bt = conv_to_BT(ch2_file, ch13_file, ch15_file)

RGB = np.stack([ch2_bt, ch13_bt, ch15_bt], axis = -1)
p_RGB = RGB[0:3500,0:3500, :]
print(f"shape of the image is : {p_RGB.shape} ")

# the input is the path to the glm file I sent to you.
fp = get_pof(glmf)

rgb_uint8 = (p_RGB * 255) .astype(np.uint8)
plt.figure(figsize=(7, 2.5), dpi=400)
plt.rcParams.update({'font.size': 6})
plt.subplots_adjust(wspace=0.5)
# hmap = np.load("/data/s2896370/MobileNetV2/hmap-mm-2.npy")


ax1 = plt.subplot(1,3,1)
# plt.scatter(fp[:,1], fp[:,0], cmap="gray",marker="s", s=5)
plt.title("Original Image", fontsize=7)
pixel_plot = plt.imshow(rgb_uint8, cmap='twilight', interpolation='nearest')
plt.colorbar(pixel_plot,fraction=0.046, pad=0.04)

ax2 = plt.subplot(1,3,2)
plt.scatter(fp[:,1], fp[:,0], c="lightgreen", marker="s", s=5, edgecolors="black", linewidths=0.1)
hmap = np.load("/data/s2896370/MobileNetV2/Source/np_heatmaps/hmap-mm-2.npy")
plt.title("MM Heatmap", fontsize=7)
pixel_plot = plt.imshow(hmap, cmap='hot', interpolation='nearest')
plt.colorbar(pixel_plot,fraction=0.046, pad=0.04)


ax3 = plt.subplot(1,3,3)
# path to the heatmap file(hmap_...)
hmap_res = np.load(hmap_path)
# hmap_res = most_prob_circle(hmap_res)
plt.title("Prediction Heatmap SimpleCNN", fontsize=7)
plt.scatter(fp[:,1], fp[:,0], c="lightgreen", marker="s", s=5, edgecolors="black", linewidths=0.1)
plot = plt.imshow(hmap_res, cmap='hot', interpolation='nearest')
plt.colorbar(plot,fraction=0.046, pad=0.04)
# plt.suptitle("year "+ year + "day " + day + "hour " + hour)
plt.savefig("/data/s2896370/MobileNetV2/final_heatmaps/simp_org_35min.png")
