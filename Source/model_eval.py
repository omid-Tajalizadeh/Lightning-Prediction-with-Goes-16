import tensorflow as tf
from keras.optimizers import RMSprop,Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras



lr= 0.0001
model = tf.keras.models.Sequential([
    # since Conv2D is the first layer of the neural network, we should also specify the size of the input
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    # apply pooling
    tf.keras.layers.MaxPooling2D(2, 2),
    # and repeat the process
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # flatten the result to feed it to the dense layer
    tf.keras.layers.Flatten(),
    # and define 512 neurons for processing the output coming by the previous layers
    tf.keras.layers.Dense(512, activation='relu'),
    # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
    tf.keras.layers.Dense(1, activation='sigmoid')
])

opt = Adam(learning_rate=lr)
# earlystop = EarlyStopping(monitor='val_accuracy', patience=20, verbose=0, mode='auto')
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


model.load_weights("model/weights.h5")
model.summary()
datagen = ImageDataGenerator()
val_it = datagen.flow_from_directory('data64/validation/', class_mode='binary',target_size=(64, 64))


loss, accuracy = model.evaluate(val_it)
print(loss, accuracy)