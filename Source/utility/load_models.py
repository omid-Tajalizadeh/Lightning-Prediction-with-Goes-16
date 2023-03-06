import tensorflow as tf 
from keras.optimizers import RMSprop,Adam
from keras.models import Model
import os
import tempfile
from keras import regularizers

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

def load_model(name, size, lr):
    if name == "res":
        model = tf.keras.applications.ResNet50(include_top=True,
        input_shape=(size,size,3),
        weights=None, 
        classes=1,
        classifier_activation='sigmoid')
        model = add_regularization(model, tf.keras.regularizers.L1(0.0001))
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])

    elif name == "simple":
        model = tf.keras.models.Sequential([ 
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(size, size, 3), kernel_regularizer=regularizers.l2(0.0001)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu',  kernel_regularizer=regularizers.l2(0.0001)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])


    elif name == "mnv3s":
        model = tf.keras.applications.MobileNetV3Small(include_top=True,
        weights=None,
        input_shape=(size,size,3),
        classes=1, 
        dropout_rate=0.2, 
        classifier_activation='sigmoid', 
        alpha = 1.0)
        model = add_regularization(model, tf.keras.regularizers.L1(0.0001))

        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])


    elif name == "mnv3l":
        model = tf.keras.applications.MobileNetV3Large(include_top=True,
        weights=None,
        input_shape=(size,size,3),
        classes=1, 
        dropout_rate=0.2, 
        classifier_activation='sigmoid', 
        alpha = 0.5)
        model = add_regularization(model, tf.keras.regularizers.L1(0.0001))

        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])
    
    elif name == "dense":
        model = tf.keras.applications.densenet.DenseNet121(include_top=True,
        weights=None,
        input_shape=(size,size,3),
        classes=1,  
        classifier_activation='sigmoid')
        model = add_regularization(model, tf.keras.regularizers.L1L2(0.0002))
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])

    return model