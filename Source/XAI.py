import numpy as np
import shap
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image

# import lime
from utility.load_models import * 
from lime import lime_image

# Load the pre-trained ResNet50 model

# Load the image and preprocess it



"""
This code uses the pre-trained ResNet50 model from the Keras library,
which is a deep convolutional neural network trained on the ImageNet dataset. 
It loads a subset of the ImageNet dataset, and then creates a LimeImageExplainer 
object to explain the predictions of the model on the first 10 images in the dataset. 
The explain_instance function is used to generate the explanations, and the top_labels
argument specifies the number of labels to display in the explanation. 
The hide_color argument is used to hide the original image, 
and the num_samples argument specifies the number of randomly generated perturbations of the input data to use 
when generating the explanation. Finally, the explanation for the first image is plotted as a figure.
"""


def lime(model, x):



    # Load the pre-trained ResNet model
    x = np.expand_dims(x, axis=0)

    # Get the model prediction
    pred = model.predict(x)
    pred_class = int(round(pred[0][0]))
    print(pred_class)

    # Define the LimeImageExplainer
    explainer = lime_image.LimeImageExplainer()

    # Define the prediction function for the model
    def predict_fn(images):
        preds = model.predict(images)
        return preds

    # Explain the prediction using LIME
    explanation = explainer.explain_instance(x[0], predict_fn, top_labels=2, hide_color=0, num_samples=1000)

    # Show the explanation
    temp, mask = explanation.get_image_and_mask(pred_class, positive_only=False, num_features=10, hide_rest=False)
    img_lime = mark_boundaries(temp / 2 + 0.5, mask)
    img = Image.fromarray((img_lime * 255).astype(np.uint8))
    print(img)




"""
This code uses the pre-trained ResNet50 model from the Keras library,
which is a deep convolutional neural network trained on the ImageNet dataset. 
It loads a subset of the ImageNet dataset using the shap library, 
computes the SHAP values for the model's predictions on the first 10 images in the dataset, 
and then plots a waterfall plot of the SHAP values for the first image. 
The SHAP values represent the contribution of each feature in the input data to the model's prediction.

Note that the Explainer class from the shap library is used to wrap the model's
prediction function and make it compatible with the library's functions.
The nsamples argument in the explainer function specifies the number of
Monte Carlo samples to use when estimating the SHAP values.
"""

    
class_names = ["NoFlash", "Flash"]

def shap_values(model, X, y, class_names={}):
    def f(x):
        tmp = x.copy()
        preprocess_input(tmp)
        return model(tmp)
    
    
    masker = shap.maskers.Image("blur(128,128)", X[0].shape)
    explainer = shap.Explainer(f, masker)
    shap_val = explainer(X[1:5], max_evals=3000, batch_size=100)
    

# # Visualize the SHAP values for the R channel
#     shap.image_plot(shap_values_R, x[:, :, 0])

# # Visualize the SHAP values for the G channel
#     shap.image_plot(shap_values_G, x[:, :, 1])

# # Visualize the SHAP values for the B channel
    # Shap.image_plot(shap_values_B, x[:, :, 2])

    return shap_val


size = 64
lr = 0.0000001




model = load_model("simple", size=size, lr = lr)
# model.load_weights("/data/s2896370/MobileNetV2/ResNet50_35mins/weights.h5")
model.load_weights("/data/s2896370/MobileNetV2/Source/SimpleCNN_35min/weights.h5")

model.summary()

X = np.load("/data/s2896370/MobileNetV2/cut_npy/35min/test_35min.npy")
y = np.load("/data/s2896370/MobileNetV2/cut_npy/35min/test_labels_35min.npy")

lime(model, X[0])


# # val = shap_values(model, X, y)

# # shap.image_plot(val.values)
# def predict_fn(img):
#     img = tf.image.resize(img, (64, 64)) / 255.0
#     return model(img)

# x = X[None,0,...]

# background = np.zeros((10, 64,64,3), dtype=np.float32)

# # Create an explainer object using the DeepExplainer class
# explainer = shap.DeepExplainer(model, background)
# s_values = explainer.shap_values(x)
# Calculate SHAP values for the R channel
# shap_values_R = explainer.shap_values(x[:, :, 0], ranked_outputs=1)

# # Calculate SHAP values for the G channel
# shap_values_G = explainer.shap_values(x[:, :, 1], ranked_outputs=1)

# # Calculate SHAP values for the B channel
# shap_values_B = explainer.shap_values(x[:, :, 2], ranked_outputs=1)

# # Visualize the SHAP values for the R channel
# shap.image_plot(shap_values_R, x[:, :, 0])

# # Visualize the SHAP values for the G channel
# shap.image_plot(shap_values_G, x[:, :, 1])

# # Visualize the SHAP values for the B channel
# shap.image_plot(shap_values_B, x[:, :, 2])
# np.save("Shap/shap_val_res", val)
# Load an example image


# Preprocess the image for prediction
# x = tf.keras.preprocessing.image.img_to_array(img)

