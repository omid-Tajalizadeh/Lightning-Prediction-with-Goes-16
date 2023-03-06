from netCDF4 import Dataset as ncds
import numpy as np
import copy
import cv2
import tempfile

from patchify import patchify
import tensorflow as tf
from keras import regularizers
import os
from keras.optimizers import RMSprop,Adam



import matplotlib.pyplot as plt
import random


from utility.load_models import *
from utility.flash_pairs import *
from utility.to_BT import conv_to_BT 




# ch2_file = "/data/s2896370/MobileNetV2/3ch/1/OR_ABI-L1b-RadF-M6C02_G16_s20191081010223_e20191081019531_c20191081019570.nc"
# ch13_file = "/data/s2896370/MobileNetV2/3ch/1/OR_ABI-L1b-RadF-M6C13_G16_s20191081010223_e20191081019542_c20191081020004.nc"
# ch15_file = "/data/s2896370/MobileNetV2/3ch/1/OR_ABI-L1b-RadF-M6C15_G16_s20191081010223_e20191081019537_c20191081020003.nc"
# glmf = "/data/s2896370/MobileNetV2/3ch/1/GLM-L2-LCFA.noaa-goes16.20191081010.CaribbeanLarge.nc"


ch2_file = "/data/s2896370/MobileNetV2/3ch/2/OR_ABI-L1b-RadF-M6C02_G16_s20192631500194_e20192631509502_c20192631509546.nc"
ch13_file = "/data/s2896370/MobileNetV2/3ch/2/OR_ABI-L1b-RadF-M6C13_G16_s20192631500194_e20192631509513_c20192631510000.nc"
ch15_file = "/data/s2896370/MobileNetV2/3ch/2/OR_ABI-L1b-RadF-M6C15_G16_s20192631500194_e20192631509508_c20192631509594.nc"
glmf = "/data/s2896370/MobileNetV2/3ch/2/GLM-L2-LCFA.noaa-goes16.20192631500.CaribbeanLarge.nc"


day =  ch2_file.split("_")[3][5:8]
year =  ch2_file.split("_")[3][1:5]
hour = ch2_file.split("_")[3][8:10]

ch2_normalized, ch13_resized, ch15_resized = conv_to_BT(ch2_file, ch13_file, ch15_file)


def add_regularization(model, regularizer):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def max_mean(img):
    for i in range(img.shape[0]):
        b = np.max(img[:,:,0])
        c = np.max(img[:,:,1])
        d = np.max(img[:,:,2])
        
    return np.mean([b,c,d]) 

    


RGB = np.stack([ch2_normalized, ch13_resized, ch15_resized], axis = -1)
p_RGB = RGB[0:3500,0:3500, :]
print(f"shape of the image is : {p_RGB.shape} ")

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
patches =  patchify(p_RGB, (64,64,3), step=1)
with strategy.scope():

    hmap_res = np.zeros((patches.shape[0],patches.shape[1]))


    # lr = 0.0000001
    lr = 0.00000001
    model = load_model("mnv3s", 64, lr=lr)
    model.load_weights("/data/s2896370/MobileNetV2/Source/MobileNetV3_35min/weights.h5")
    # model.load_weights("/data/s2896370/MobileNetV2/Source/DenseNet121_35min/weights.h5")
    # model.load_weights("/data/s2896370/MobileNetV2/ResNet50_35mins/weights.h5")
    # model.load_weights("/data/s2896370/MobileNetV2/Source/MobileNetV3Large_35min/weights.h5")
    
    model.summary()
    for i in range(patches.shape[0]):
        hmap_res[i] = model.predict(patches[i,:,0,:,:,:], batch_size=30)[:,0]
    # np.save("hmap-res-170mins.npy", hmap_res)
    np.save("hmap-mnetS-35mins.npy", hmap_res)


hmap = np.zeros((patches.shape[0],patches.shape[1]))

# print("Creating Mean-max Heatmap ----------------:")
# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         hmap[i][j] = max_mean(patches[i,j,0,:,:,:])
# print("Map creation done!")


# fp = get_pof(glmf)

# # hmap = np.array(hmap)
# # np.save("hmap-mm-2.npy", hmap)

# rgb_uint8 = (p_RGB * 255) .astype(np.uint8)
# plt.subplots_adjust(hspace=0.45, wspace=0.45)
# plt.subplot(2,2,1)
# plt.scatter(fp[:,1], fp[:,0], c="yellow",marker="s", s=5)
# plt.title("Original Image")

# pixel_plot = plt.imshow(rgb_uint8, cmap='gray', interpolation='nearest')
# plt.colorbar(pixel_plot,fraction=0.046, pad=0.04)


# plt.subplot(2,2,2)
# plt.title("Prediction Heatmap MM")
# # hmap = np.load("/data/s2896370/MobileNetV2/hmap-mm.npy")
# hmap = np.load("/data/s2896370/MobileNetV2/hmap-mm-2.npy")
# plt.scatter(fp[:,1], fp[:,0], c="black",marker="s", s=5)
# plot = plt.imshow(hmap, cmap='hot', interpolation='nearest')
# plt.colorbar(plot,fraction=0.046, pad=0.04)




# plt.subplot(2,2,3)
# # hmap_res = np.load("/data/s2896370/MobileNetV2/hmap-res-35mins.npy")
# plt.title("Prediction Heatmap DenseNet121")
# plot = plt.imshow(hmap_res, cmap='hot', interpolation='nearest')
# plt.scatter(fp[:,1], fp[:,0], c="paleturquoise", marker="s", s=5)
# plt.colorbar(plot)
# # plt.clim(0,1)

# plt.suptitle("year "+ year + "day " + day + "hour " + hour)

# plt.savefig("hmap_dense_35min.png")
# plt.savefig("resn_mm_hmap2.png")