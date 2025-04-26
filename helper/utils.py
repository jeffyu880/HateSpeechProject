import os 
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import matplotlib.pyplot as plt

def calculate_mmae(expected, predicted, classes):
    NUM_CLASSES = len(classes)
    count_dict = {}
    dist_dict = {}
    for i in range(NUM_CLASSES):
        count_dict[i] = 0
        dist_dict[i] = 0.0
    for i in range(len(expected)):
        dist_dict[expected[i]] += abs(expected[i] - predicted[i])
        count_dict[expected[i]] += 1
    overall = 0.0
    for claz in range(NUM_CLASSES): 
        class_dist =  1.0 * dist_dict[claz] / count_dict[claz] 
        overall += class_dist
    overall /= NUM_CLASSES
#     return overall[0]
    return overall

def plot_curves(train_acc_list, val_acc_list, train_loss_list, val_loss_list, results_path):

    epochs = range(len(train_acc_list))
    train_acc_list = [t.cpu() for t in train_acc_list]
    val_acc_list = [t.cpu() for t in val_acc_list]
    # train_loss_list = [t.cpu().numpy() for t in train_loss_list]
    # val_loss_list = [t.cpu().numpy() for t in val_loss_list]
    # plt.plot(epochs, train_acc_list)
    # plt.plot(epochs, val_acc_list)
    # print("Train ACC List: ", train_acc_list)
    # print("\nVal acc list: ", val_acc_list, 
    #     "\ntrain loss list: ", train_loss_list, 
    #     "\nval loss list: ", val_loss_list)
    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, train_acc_list, label="train acc")
    ax1.plot(epochs, val_acc_list, label="val acc")
    ax1.set_title("Accuracy Plot")
    ax1.set_xlabel("Epochs")
    ax1.legend(loc="upper left")
    acc_filename = os.path.join(results_path, f"accuracy_plot.png")
    fig1.savefig(acc_filename)

    # Loss Plot
    fig2, ax2 = plt.subplots()
    ax2.plot(epochs, train_loss_list, label="train loss")
    ax2.plot(epochs, val_loss_list, label="val loss")
    ax2.set_title("Loss Plot")
    ax2.set_xlabel("Epochs")
    ax2.legend(loc="upper left")
    loss_filename = os.path.join(results_path, f"loss_plot.png")
    fig2.savefig(loss_filename)