
# import tensorflow as tf
from netCDF4 import Dataset as ncds
import numpy as np
import copy
import cv2
from patchify import patchify
import tensorflow as tf
from keras import regularizers
import os
from keras.optimizers import RMSprop,Adam
import tempfile
import matplotlib.pyplot as plt
import random


# def create_model(l1, input_shape):
#     model = tf.keras.models.Sequential([ 
#         tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], input_shape[2]), kernel_regularizer = tf.keras.regularizers.L1(l1)),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu' , kernel_regularizer = tf.keras.regularizers.L1(l1)),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer = tf.keras.regularizers.L1(l1)),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
    
#     return model
ch2_file = "/data/s2896370/MobileNetV2/3ch/OR_ABI-L1b-RadF-M6C02_G16_s20191081010223_e20191081019531_c20191081019570.nc"
ch13_file = "/data/s2896370/MobileNetV2/3ch/OR_ABI-L1b-RadF-M6C13_G16_s20191081010223_e20191081019542_c20191081020004.nc"
ch15_file = "/data/s2896370/MobileNetV2/3ch/OR_ABI-L1b-RadF-M6C15_G16_s20191081010223_e20191081019537_c20191081020003.nc"
ch2 =  ncds(ch2_file)
ch13 = ncds(ch13_file)
ch15 = ncds(ch15_file)

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

    

ch2_normalized = ch2_normalized(ch2)
ch13_normalized = convert_to_normBT(ch13)
ch15_normalized = convert_to_normBT(ch15)

ch13_resized = cv2.resize(ch13_normalized, (6201,3151), interpolation = cv2.INTER_AREA)
ch15_resized = cv2.resize(ch15_normalized, (6201,3151), interpolation = cv2.INTER_AREA)

RGB = np.stack([ch2_normalized, ch13_resized, ch15_resized], axis = -1)

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    patches =  patchify(RGB, (64,64,3), step=1) 
    hmap = np.zeros((patches.shape[0],patches.shape[1]))

    base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(64,64,3),weights=None)
    base_model = add_regularization(base_model, regularizer=regularizers.L1L2(0.0001))
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.GlobalMaxPooling2D())
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.layers[0].trainable = True

    lr = 0.0000001
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights("/data/s2896370/MobileNetV2/ResNet50(CV)/weights_Fold3.h5")
    model.summary()
    for i in range(patches.shape[0]):
        hmap[i] = model.predict(patches[0][:,0,:,:], batch_size=100)[:,0]
            
    # print(patches[0].shape)

    # hmap = model.predict(patches, batch_size=1000)
    # for i in range(patches.shape[0]):
    #     for j in range(patches.shape[1]):

    #         hmap[i][j] = model.predict(patches[i][j])


    rgb_uint8 = (RGB * 255) .astype(np.uint8)
    pixel_plot = plt.figure()
    pixel_plot.add_axes()
    pixel_plot = plt.imshow(rgb_uint8, cmap='twilight', interpolation='nearest')
    plt.colorbar(pixel_plot)
    plt.savefig("mainpic.png")
    plt.show()
    plt.close()



    plot = plt.imshow(hmap, cmap='hot', interpolation='nearest')
    plt.colorbar(plot)
    plt.savefig("pred-hm.png")
    plt.show()
    plt.close()

    # path_pos = 'cut_npy/pos_test_35min.npy'

    # path_neg  = 'cut_npy/neg_test_35min.npy'
    # a = np.load(path_pos)
    # p_labels = np.ones(a.shape[0])
    # b = np.load(path_neg)
    # n_labels = np.zeros(b.shape[0])
    # c = list(zip(a, p_labels))
    # c1 = list(zip(b, n_labels))

    # comp = c + c1

    # random.shuffle(comp)

    # X, y = zip(*comp)

    # np.save('cut_npy/test_35min.npy', np.array(X))
    # np.save('cut_npy/test_labels_35min.npy', np.array(y))




