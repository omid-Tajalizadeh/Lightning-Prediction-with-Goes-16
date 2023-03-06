import numpy as np 
import cv2 
import matplotlib.pyplot as plt

def most_prob_circle(hmap):

    hmap_res = (hmap * 255) .astype(np.uint8)

    image = hmap_res
    orig = image.copy()

    # cv2.circle(image, maxLoc, 300, (255, 0, 0), 5)
    # # display the results of the naive attempt
    # cv2.imshow("Naive", image)
    # image = cv2.GaussianBlur(image, (300, 300), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
    circ  = cv2.circle(image, maxLoc, 500, (255, 0, 0), 7)
    return circ