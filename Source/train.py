
import os
import sys
import argparse
import pandas as pd
import numpy as np
import random

from utility.load_models import *


import tensorflow as tf

from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Reshape, Activation
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import KFold

import tempfile
import matplotlib.pyplot as plt

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
   
time_int = "35min"



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        default=64,
        help="image size.")
    parser.add_argument(
        "--batch",
        default=32,
        help="train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=100,
        help="number of epochs.")


    args = parser.parse_args()

    train(int(args.batch), int(args.epochs), int(args.size))
    
    
    
    
# function for loading the train and test data
def load_tt_data(batch):
      

    X_train = np.load("/data/s2896370/MobileNetV2/cut_npy/" + time_int + "/train_"+ time_int +".npy")
    # X_train = np.load("cut_npy/noisy_data/noisy_train_35min.npy")
   
    y_train = np.load("/data/s2896370/MobileNetV2/cut_npy/"+ time_int+ "/train_labels_"+ time_int + ".npy")


    X_test = np.load("/data/s2896370/MobileNetV2/cut_npy/"+ time_int + "/test_"+ time_int + ".npy")
    # X_test = np.load("cut_npy/noisy_data/noisy_test_35min.npy")

 
    y_test = np.load("/data/s2896370/MobileNetV2/cut_npy/" + time_int + "/test_labels_" + time_int + ".npy")
    


    # print(f"X_train: {X_train.shape}")
    # print(f"X_test: {X_test.shape}")
    # print(f"y_train: {y_train.shape}")
    # print(f"y_test: {y_test.shape}")


    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = batch
    SHUFFLE_BUFFER_SIZE = 100

    train_ds = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    # val_ds = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = test_dataset.batch(BATCH_SIZE)


    return train_ds, val_ds

#Function to load the full data(in case of cross validation)
def load_full_data(pos_path, neg_path):

    p = np.load(pos_path)
    p_labels = np.ones(p.shape[0])
    n = np.load(neg_path)
    n_labels = np.zeros(n.shape[0])
    c = list(zip(p, p_labels))
    c1 = list(zip(n, n_labels))

    comp = c+c1

    random.shuffle(comp)

    X, y = zip(*comp)
    X = np.array(X)
    y= np.array(y)

    return X,y

# function to add regularization to the keras models

def add_regularization(model, regularizer):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    model_json = model.to_json()

    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    model = tf.keras.models.model_from_json(model_json)

    model.load_weights(tmp_weights_path, by_name=True)
    return model

# scheduler for the learning rate
def scheduler(epoch, lr):

    if 10<epoch<101 and epoch%100==0:
        lr = lr + (5 * lr)

    elif epoch>100 and epoch%50==0:
        lr = lr + (5 * lr)
    
    # elif epoch>100 and epoch%40==0:
    #     lr = lr + 0.0001
        
    return lr



#plots histories of cross validation
def plot_histories(target_path, histories, metrics = ['loss', 'accuracy', 'auc', 'val_accuracy','val_loss', 'val_auc']):

    fig, axes = plt.subplots(nrows = (len(metrics) - 1) // 2 + 1, ncols = 2, figsize = (20,20))
    axes = axes.reshape((len(metrics) - 1) // 2 + 1, 2)
    for i,metric in enumerate(metrics):
        for history in histories:
            axes[(i+2)//2 - 1, 1 - (i+1)%2].plot(history[metric])
            axes[(i+2)//2 - 1, 1 - (i+1)%2].legend([i for i in range(len(histories))])
            axes[(i+2)//2 - 1, 1 - (i+1)%2].set_xticks(
                np.arange(max(history[metric]))
            )
    plt.savefig(target_path)


#plots the the metrics of the
def plot_res(hist_path, target_path):
    df = pd.read_csv(hist_path)

    plt.plot(df["loss"], color="blue", alpha=0.5)
    plt.plot(df["val_loss"], color="red", alpha=0.5)
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "validation"])
    plt.savefig(target_path + "/model_loss.png")
    plt.close()

    plt.plot(df["accuracy"], color="blue", alpha=0.5)
    plt.plot(df["val_accuracy"], color="red", alpha=0.5)
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "validation"])
    plt.savefig(target_path + "/model_accuracy.png")
    plt.close()

    plt.plot(df["auc"], color="blue", alpha=0.5)
    plt.plot(df["val_auc"], color="red", alpha=0.5)
    plt.title("model AUC scores")
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.legend(["train", "validation"])
    plt.savefig(target_path + "/model_auc.png")


def train(batch, epochs, size):

    
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
    # with tf.device("/GPU:3"):
        train_ds, val_ds = load_tt_data(batch)


        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        lr = 0.0000001 
        # num_folds = 5
        # kfold = KFold(n_splits=num_folds, shuffle=True)
        # histories = []
        # fold_num = 1
        
        # X, y = load_full_data(path_pos,path_neg)

        # for train, test in kfold.split(X, y):


        #         model_name = "VGG16"
        # model_name = "MobileNetV3Small"

        model = load_model("mnv3s", size=64, lr=lr)

        # model_name = "ResNet50"
        # model_name = "SimpleCNN"
        model_name = "MobileNetV3Small_1.0)"
        # model_name = "DenseNet121"

        # earlystop = EarlyStopping(monitor='val_accuracy', patience=100, verbose=0, mode='auto')
        # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        if not os.path.exists(model_name + "_" + time_int + "/" + str(lr)):
            os.makedirs(model_name + "_" + time_int + "/" + str(lr))


        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        # filepath=model_name+ "/" + str(lr)+ "checkpoint",
        # save_weights_only=True,
        # monitor='accuracy',
        # mode='max',
        # save_best_only=True)
        # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])

        model.summary()
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs)

        
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv(model_name + "_" + time_int + "/" + str(lr) +"/hist.csv", encoding='utf-8', index=False)
        model.save_weights(model_name + "_" + time_int + "/" + str(lr) + "/weights.h5")

            # hist = model.fit(X[train],y[train],
            # validation_data=(X[test], y[test]),
            # batch_size=batch,
            # epochs=epochs)

            # histories.append(hist)

            # df = pd.DataFrame.from_dict(hist.history)
            # df.to_csv(model_name+"/hist_Fold"+str(fold_num) +".csv", encoding='utf-8', index=False)
            


            # model.save_weights(model_name+"/weights_Fold" + str(fold_num) +".h5")
            # fold_num = fold_num + 1


        data_type = str(size)
        # t_path = "Plots/Network-Plots/" + model_name + "/" + data_type + "/lr(" + str(lr) + ")"

    
           




    history_path = model_name + "_" + time_int + "/" + str(lr) +"/hist.csv"

    
    t_path = "Plots/Network-Plots/" + model_name + "_" + time_int + "/"+ data_type + "/lr(" + str(lr) + ")"
    if not os.path.exists(t_path):
        os.makedirs(t_path)

    plot_res(history_path, t_path)



if __name__ == '__main__':
    main(sys.argv)
