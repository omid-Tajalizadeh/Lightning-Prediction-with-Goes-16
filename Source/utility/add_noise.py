import numpy as np
import skimage

train = np.load("/data/s2896370/MobileNetV2/cut_npy/train_35min.npy")
test = np.load("/data/s2896370/MobileNetV2/cut_npy/test_35min.npy")

var = 0.4
def add_gaussian_noise(X_imgs, mode):
    # Gaussian distribution parameters
    images = []

    for img in X_imgs:

        gimg = skimage.util.random_noise(img, mode=mode, var=var)
        images.append(gimg)
    gaussian_imgs = np.array(images, dtype = np.float32)
    
    return gaussian_imgs

noisy_test = add_gaussian_noise(test, "gaussian")
noisy_train = add_gaussian_noise(train, "gaussian")

np.save("/data/s2896370/MobileNetV2/cut_npy/noisy_data/noisy"+ str(var) +"_train_35min", noisy_train)
np.save("/data/s2896370/MobileNetV2/cut_npy/noisy_data/noisy"+ str(var) +"_test_35min", noisy_test)

# print(np.load("noisy_test_35min.npy").shape)