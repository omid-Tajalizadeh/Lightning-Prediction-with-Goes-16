import cv2
import numpy as np
from PIL import Image
path = "/data/s2896370/MobileNetV2/hmap-res-170mins.npy"

img = np.load(path)
rgb = (img * 255) .astype(np.uint8)
img = Image.fromarray(rgb)

img.save("new.png")
path1 = "/data/s2896370/MobileNetV2/new.png"

img1 = cv2.imread(path1,0)
equ = cv2.equalizeHist(img1)


# print(equ)
cv2.imwrite("equ.png", equ)



