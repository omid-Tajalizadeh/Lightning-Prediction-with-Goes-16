import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


labels = [["1", "Flash"], ["0", "NoFlash"]]
subsets = [["train", "Training"], ["validation", "Validation"]]

path = "data64"

nb_bins = 256


for l, name in labels:
    for s, S in subsets:
        count_r = np.zeros(nb_bins)
        count_g = np.zeros(nb_bins)
        count_b = np.zeros(nb_bins)
        img_path = path + "/" + s + "/" + l
        print(img_path)
        for file in os.listdir(img_path):
            full_path = os.path.join(img_path, file)
            image = cv2.imread(full_path)

            hist_r = np.histogram(image[:, :, 0], bins=nb_bins, range=[0, 255])
            hist_g = np.histogram(image[:, :, 1], bins=nb_bins, range=[0, 255])
            hist_b = np.histogram(image[:, :, 2], bins=nb_bins, range=[0, 255])

            count_r += hist_r[0]
            count_g += hist_g[0]
            count_b += hist_b[0]

        bins = hist_r[1]
        fig = plt.figure()
        plt.bar(bins[:-1], count_r, color='red', alpha=0.4)
        plt.bar(bins[:-1], count_g, color='green', alpha=0.4)
        plt.bar(bins[:-1], count_b, color='blue', alpha=0.4)
        plt.xlabel('Intensity Value')
        plt.ylabel('Count')
        plt.title(name + " " + S + " Histogram")
        plt.legend(['CH02', 'CH13', 'CH15'])
        plt.savefig("Plots/Histograms/" + name + "_" + s + path + ".png")
        plt.show()