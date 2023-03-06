import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_path = "/data/s2896370/MobileNetV2/Source/DenseNet121_35min/6e-08/hist.csv"
targ_path = "/data/s2896370/MobileNetV2/Source/utility/final_plots"
# norm_path = "/data/s2896370/MobileNetV2/cut_npy/train_35min.npy"

# rgb_uint8 = (p_RGB * 255) .astype(np.uint8)

def plot_res(hist_path, target_path, model):
    df = pd.read_csv(hist_path)

    # plt.plot(df["loss"], color="blue", alpha=0.5)
    # plt.plot(df["val_loss"], color="red", alpha=0.5)
    # plt.title("model loss")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.legend(["train", "validation"])
    # plt.savefig(target_path + "/model_loss.png")
    # plt.close()

    plt.plot(df["accuracy"], color="blue", alpha=0.5)
    plt.plot(df["val_accuracy"], color="red", alpha=0.5)
    plt.title(model + " accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "validation"])
    plt.savefig(target_path + "/" + model +"_accuracy.png", dpi=400)   
    plt.close()


    plt.plot(df["auc"], color="blue", alpha=0.5)
    plt.plot(df["val_auc"], color="red", alpha=0.5)
    plt.title(model + " AUC scores")
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.legend(["train", "validation"])

    plt.savefig(target_path + "/" + model +"_AUC.png", dpi=400)    
    
    
plot_res(data_path, targ_path, "DenseNet121")


# def plot_res(hist_path, target_path):
#     df = pd.read_csv(hist_path)

#     plt.plot(df["loss"], color="blue", alpha=0.5)
#     plt.plot(df["val_loss"], color="red", alpha=0.5)
#     plt.title("model loss")
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.legend(["train", "validation"])
#     plt.savefig(target_path + "/model_loss.png")
#     plt.close()

#     plt.plot(df["accuracy"], color="blue", alpha=0.5)
#     plt.plot(df["val_accuracy"], color="red", alpha=0.5)
#     plt.title("model accuracy")
#     plt.xlabel("epoch")
#     plt.ylabel("accuracy")
#     plt.legend(["train", "validation"])
#     plt.savefig(target_path + "/model_accuracy.png")
#     plt.close()

#     plt.plot(df["auc"], color="blue", alpha=0.5)
#     plt.plot(df["val_auc"], color="red", alpha=0.5)
#     plt.title("model AUC scores")
#     plt.xlabel("epoch")
#     plt.ylabel("AUC")
#     plt.legend(["train", "validation"])
#     plt.savefig(target_path + "/model_auc.png")

# plot_res("/data/s2896370/MobileNetV2/simpleCNN_35mins/1e-07/hist.csv", "/data/s2896370/MobileNetV2/Plots/Network-Plots/simpleCNN_35mins")


# print(np.load("/data/s2896370/MobileNetV2/cut_npy/pos_test_35min.npy").shape)
# print(np.load("/data/s2896370/MobileNetV2/cut_npy/train_labels_35min.npy").shape)
