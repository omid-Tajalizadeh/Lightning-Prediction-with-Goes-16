from to_BT import conv_to_BT 

#insert you file paths for each channel here
ch2_file = "/data/s2896370/MobileNetV2/3ch/1/OR_ABI-L1b-RadF-M6C02_G16_s20191081010223_e20191081019531_c20191081019570.nc"
ch13_file = "/data/s2896370/MobileNetV2/3ch/1/OR_ABI-L1b-RadF-M6C13_G16_s20191081010223_e20191081019542_c20191081020004.nc"
ch15_file = "/data/s2896370/MobileNetV2/3ch/1/OR_ABI-L1b-RadF-M6C15_G16_s20191081010223_e20191081019537_c20191081020003.nc"

ch2_bt, ch13_bt, ch15_bt = conv_to_BT(ch2_file, ch13_file, ch15_file)

print("Creating the RGB images -------------")



rgb_ch2 = (ch2_bt * 255) .astype(np.uint8)
plt.subplots_adjust(hspace=0.45, wspace=0.45)
plt.subplot(2,2,1)

plt.title("Channel 2 Image")

pixel_plot = plt.imshow(rgb_ch2, cmap='twilight', interpolation='nearest')
plt.colorbar(pixel_plot)


rgb_ch13 = (ch13_bt * 255) .astype(np.uint8)
plt.subplot(2,2,2)
plt.title("Channel 13 image")
plot = plt.imshow(rgb_ch13, cmap='twilight', interpolation='nearest')
plt.colorbar(plot)




rgb_ch15 = (ch15_bt * 255) .astype(np.uint8)
plt.subplot(2,2,3)
plt.title("Channel 15 image")
plot = plt.imshow(rgb_ch15, cmap='twilight', interpolation='nearest')
plt.colorbar(plot)
plt.suptitle(year + " day " + day + " hour " + hour)

plt.savefig("rgb.png")

print("RGB image creation complete ---------")
