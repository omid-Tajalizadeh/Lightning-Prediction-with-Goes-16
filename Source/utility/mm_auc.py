import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

import matplotlib.pyplot as plt




# X_test = np.load("/data/s2896370/MobileNetV2/cut_npy/test_35min.npy")
# y_test = np.load("/data/s2896370/MobileNetV2/cut_npy/test_labels_35min.npy")

def auc_probability(y_true, y_score, size):
    """probabilistic interpretation of AUC"""
    labels = y_true.astype(bool)
    pos = np.random.choice(y_score[labels], size = size, replace = True)
    neg = np.random.choice(y_score[~labels], size = size, replace = True)
    auc = np.sum(pos > neg) + np.sum(pos == neg) / 2
    auc /= size
    return auc

def max_mean(img):
    for i in range(img.shape[0]):
        b = np.max(img[:,:,0])
        c = np.max(img[:,:,1])
        d = np.max(img[:,:,2])
        
    return np.mean([b,c,d])


def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """
    
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.title("SimpleCNN Heatmap ROC Curve", fontsize=11)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("roc_mm.png")


# p_yt = y_test[0:3000]
# p_xt = X_test[0:3000]


# y_score = []
# for img in X_test:
#     y_score.append(max_mean(img))
y_true = np.load("/data/s2896370/MobileNetV2/Source/utility/patch_labels/patch_labels_simp.npy")
y_true = y_true.flatten()

print(y_true.shape)

y_score = np.load("/data/s2896370/MobileNetV2/Source/hmap-dense-35mins.npy")
y_score = y_score.flatten()

print(y_score.shape)

# print(y_test.shape[0])

print(auc_probability(y_true, y_score, size = y_true.shape[0]))

# print(roc_auc_score(y_test, y_score))

plot_roc_curve(y_true, y_score)





