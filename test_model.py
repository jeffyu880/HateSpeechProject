# Import all the dependencies
import torch 
import torchvision
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# from torchnlp import encoders
# from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import  mean_absolute_error
# from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from pathlib import Path
# import matplotlib.pyplot as plt
from early_stopping_pytorch import EarlyStopping
from sentence_transformers import SentenceTransformer
import torch
from torch import optim, nn
from torchvision import models, transforms
import cv2
# import gzip
import os
# from functools import lru_cache
# import ftfy
# import regex as re
import wandb

# import additional files
import helper.datasetaug as DatasetAugmentation
from helper.config import config
from helper.model import MM
from helper.utils import *

# Evaluate on test-set
outputs = test_model(model, criterion)

# # Binary setting
np_out = np.array(outputs)
y_pred = np.zeros(np_out.shape)
y_pred[np_out>0.5]=1
y_pred = np.array(y_pred)

# # Binary setting
test_labels=[]
# for index, row in test_samples_frame.iterrows():
for index, row in test_samples_frame.iterrows():
    lab = row['labels'][0]
    if lab=="not harmful":
        test_labels.append(0)    
    else:
        test_labels.append(1)

rec = np.round(recall_score(test_labels, y_pred, average="macro"),4)
prec = np.round(precision_score(test_labels, y_pred, average="macro"),4)
f1 = np.round(f1_score(test_labels, y_pred, average="macro"),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc = np.round(accuracy_score(test_labels, y_pred),4)
mmae = np.round(calculate_mmae(test_labels, y_pred, [0,1]),4)
mae = np.round(mean_absolute_error(test_labels, y_pred),4)
# print("recall_score\t: ",rec)
# print("precision_score\t: ",prec)
# print("f1_score\t: ",f1)
# print("hamming_loss\t: ",hl)
# print("accuracy_score\t: ",f1)
print(classification_report(test_labels, y_pred))
print("Acc, F1, Rec, Prec, MAE, MMAE")
print(acc, f1, rec, prec, mae, mmae)