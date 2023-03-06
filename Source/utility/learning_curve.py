from load_models import *


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from keras.models import Model


X_train = np.load("/data/s2896370/MobileNetV2/cut_npy/170min/train_170min.npy")
y_train = np.load("/data/s2896370/MobileNetV2/cut_npy/170min/train_labels_170min.npy")


X_test = np.load("/data/s2896370/MobileNetV2/cut_npy/170min/test_170min.npy")
y_test = np.load("/data/s2896370/MobileNetV2/cut_npy/170min/test_labels_170min.npy")    

lr1 = 0.00000005






powers_of_two = np.power(2, np.arange(7))
hun = 100
result = powers_of_two[powers_of_two <= 100]
result = np.append(result, hun)
train_sizes = result * 0.01
train_scores = []
test_scores = []

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():

    model = load_model("Resnet50", size =64, lr=lr1)
    for train_size in train_sizes:
        print(f"training the model with {train_size} train dataset ---------------------")
        X_train_subset = X_train[:int(train_size * X_train.shape[0])]
        y_train_subset = y_train[:int(train_size * y_train.shape[0])]

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_subset, y_train_subset))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        BATCH_SIZE = 10
        SHUFFLE_BUFFER_SIZE = 100

        
        train_ds = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        val_ds = test_dataset.batch(BATCH_SIZE)

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        model.fit(train_ds, epochs=5, batch_size=100)
        
        train_scores.append(model.evaluate(train_ds, verbose=0)[1])
        test_scores.append(model.evaluate(val_ds, verbose=0)[1])


# Plot the learning curve
plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)

plt.grid()

plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training Score")
plt.plot(train_sizes, test_scores, 'o-', color="g", label="Testing Score")

plt.legend(loc="best")
plt.savefig("learning_curve_170minsfg.png")
