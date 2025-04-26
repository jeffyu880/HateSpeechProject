#!/usr/bin/env python
# coding: utf-8

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


# wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# run = wandb.init(
#     project=config["exp_name"],  # Specify your project
#     config={                        # Track hyperparameters and metadata
#         "learning_rate": config["lr"],
#         "epochs": config["n_epochs"],
#         "batch_size": config["batch_size"]
#     },
# )


torch.cuda.empty_cache()


# COVID DATA

# Either load the pre-saved ROI/Entity features, or compute them on demand.
# Load the training, validation and test sets for the corresponding experimental setup as per the requirement

# # Load the ROI features (Covid)
# train_ROI = torch.load("path_to_features/harmeme_cov_train_ROI.pt")
# val_ROI = torch.load("path_to_features/harmeme_cov_val_ROI.pt")
# # test_ROI = torch.load("path_to_features/harmeme_cov_test_ROI.pt")
# # Load the ENT features
# train_ENT = torch.load("path_to_features/harmeme_cov_train_ent.pt")
# val_ENT = torch.load("path_to_features/harmeme_cov_val_ent.pt")
# # test_ENT = torch.load("path_to_features/harmeme_cov_test_ent.pt")


# Harmful Meme dataset (Covid-Ternary data)
# data_dir_cov = "path_to_images/images"
# train_path_cov = "path_to_jsonl/train.jsonl"
# dev_path_cov   = "path_to_jsonl/val.jsonl"
# test_path_cov  = "path_to_jsonl/test.jsonl"


# train_samples_frame = pd.read_json(train_path_cov, lines=True)
# train_samples_frame.head()

# test_samples_frame = pd.read_json(test_path_cov, lines=True)
# test_samples_frame.head()


########################## POLITICAL DATA ###############################

# # # Load the ROI features (Political)
train_ROI_path = torch.load("harmeme_saved_feat_ROIENT/harmeme_ROI_MOMENTA/pol/harmfulness/harmeme_pol_train_ROI.pt")#, map_location=torch.device('cpu'))
val_ROI_path = torch.load("harmeme_saved_feat_ROIENT/harmeme_ROI_MOMENTA/pol/harmfulness//harmeme_pol_val_ROI.pt") #, map_location=torch.device('cpu'))
test_ROI_path = torch.load("harmeme_saved_feat_ROIENT/harmeme_ROI_MOMENTA/pol/harmfulness//harmeme_pol_test_ROI.pt") #, map_location=torch.device('cpu'))
# # # Load the ENT features
train_ENT_path = torch.load("harmeme_saved_feat_ROIENT/harmeme_ENT_MOMENTA/pol/harmeme_pol_harmfulness/harmeme_pol_train_ent.pt") #, map_location=torch.device('cpu'))
val_ENT_path = torch.load("harmeme_saved_feat_ROIENT/harmeme_ENT_MOMENTA/pol/harmeme_pol_harmfulness/harmeme_pol_val_ent.pt") #, map_location=torch.device('cpu'))
test_ENT_path = torch.load("harmeme_saved_feat_ROIENT/harmeme_ENT_MOMENTA/pol/harmeme_pol_harmfulness/harmeme_pol_test_ent.pt") #, map_location=torch.device('cpu'))

# Harmful Meme dataset (Political-Binary)
data_dir_pol = '../HarMeme_Images/harmeme_images_us_pol'
train_path_pol = 'HarMeme_V1/Annotations/Harm-P/train_v1.jsonl'
dev_path_pol = "HarMeme_V1/Annotations/Harm-P/val_v1.jsonl"
test_path_pol = "HarMeme_V1/Annotations/Harm-P/test_v1.jsonl"

train_samples_frame = pd.read_json(train_path_pol, lines=True)
print("Train samples: ", train_samples_frame.head())

test_samples_frame = pd.read_json(test_path_pol, lines=True)
print("Test samples: ", test_samples_frame.head())

clip_model = torch.jit.load("pretrained/ViT-B-32.pt").cuda().eval()
# clip_model = torch.jit.load("ViT-B-32.pt").eval()
input_resolution = clip_model.input_resolution.item()
context_length = clip_model.context_length.item()
vocab_size = clip_model.vocab_size.item()

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

# ## Harmeme dataset ROI+ENT augmentation all return (both one-hot and numeric)


# hm_dataset_train = HarmemeMemesDatasetAug2(train_path_cov, data_dir_cov, 'train')
# dataloader_train = DataLoader(hm_dataset_train, batch_size=64,
#                         shuffle=True, num_workers=0)
# hm_dataset_val = HarmemeMemesDatasetAug2(dev_path_cov, data_dir_cov, 'val')
# dataloader_val = DataLoader(hm_dataset_val, batch_size=64,
#                         shuffle=True, num_workers=0)
# hm_dataset_test = HarmemeMemesDatasetAug2(test_path_pol, data_dir_pol, 'test')
# dataloader_test = DataLoader(hm_dataset_test, batch_size=64,
#                         shuffle=False, num_workers=0)


hm_dataset_train = DatasetAugmentation.HarmemeMemesDatasetAug2(data_path=train_path_pol, 
                                                               img_dir=data_dir_pol, 
                                                               train_ROI_path=train_ROI_path,
                                                               val_ROI_path=val_ROI_path,
                                                               test_ROI_path=test_ROI_path,
                                                               train_ENT_path=train_ENT_path,
                                                               val_ENT_path=val_ENT_path,
                                                               test_ENT_path=test_ENT_path,
                                                               input_resolution=input_resolution,
                                                               context_length=context_length,
                                                               clip_model=clip_model,
                                                               split_flag='train')
dataloader_train = DataLoader(hm_dataset_train, batch_size=config["batch_size"],
                        shuffle=True, num_workers=0)
hm_dataset_val = DatasetAugmentation.HarmemeMemesDatasetAug2(data_path=dev_path_pol, 
                                                             img_dir=data_dir_pol, 
                                                             train_ROI_path=train_ROI_path,
                                                             val_ROI_path=val_ROI_path,
                                                             test_ROI_path=test_ROI_path,
                                                             train_ENT_path=train_ENT_path,
                                                             val_ENT_path=val_ENT_path,
                                                             test_ENT_path=test_ENT_path,
                                                             input_resolution=input_resolution,
                                                             context_length=context_length,
                                                             clip_model=clip_model,
                                                             split_flag='val')
dataloader_val = DataLoader(hm_dataset_val, batch_size=config["batch_size"],
                        shuffle=True, num_workers=0)
hm_dataset_test = DatasetAugmentation.HarmemeMemesDatasetAug2(data_path=test_path_pol, 
                                                              img_dir=data_dir_pol, 
                                                              train_ROI_path=train_ROI_path,
                                                              val_ROI_path=val_ROI_path,
                                                              test_ROI_path=test_ROI_path,
                                                              train_ENT_path=train_ENT_path,
                                                              val_ENT_path=val_ENT_path,
                                                              test_ENT_path=test_ENT_path,
                                                              input_resolution=input_resolution,
                                                              context_length=context_length,
                                                              clip_model=clip_model,
                                                              split_flag='test')
dataloader_test = DataLoader(hm_dataset_test, batch_size=config["batch_size"],
                        shuffle=False, num_workers=0)


# ### MODEL

# Provide the specific model definition/module here

# Get the cross attention value features 
# Vanilla model

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total trainable parameters are: {total_params}")


# ## Training

# For BCE loss all features
def train_model(model, patience, n_epochs, use_wandb=True):
    epochs = n_epochs
#     clip = 5

    train_acc_list=[]
    val_acc_list=[]
    train_loss_list=[]
    val_loss_list=[]
    
    # initialize early_stopping object
    chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)
    num_batches = len(dataloader_train)

    model.train()
    for i in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for batch_idx, data in enumerate(dataloader_train):
            print(f"Epoch: {i}, Batch: {batch_idx + 1}/{num_batches}")
#             Clip features...
            img_inp_clip = data['image_clip_input']
            txt_inp_clip = data['text_clip_input']
            with torch.no_grad():
                img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)

            img_feat_vgg = data['image_vgg_feature']
            txt_feat_trans = data['text_drob_embedding']

            label = data['label'].to(device)

            model.zero_grad(), 
            output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
#             output = model(img_feat_vgg, txt_feat_trans)

            loss = criterion(output.squeeze(), label.float())
            
#             print(loss)
            loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            with torch.no_grad():
#                 print(output.squeeze().shape)
#                 print(label.float().shape)
                acc = torch.abs(output.squeeze() - label.float()).view(-1)
                acc = (1. - acc.sum() / acc.size()[0])
                total_acc_train += acc
                total_loss_train += loss.item()

        train_acc = total_acc_train/len(dataloader_train)
        train_loss = total_loss_train/len(dataloader_train)

        # wandb.log({"train accuracy: ": train_acc, "train loss: ": train_loss})

        model.eval()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for data in dataloader_val:                
#                 Clip features...                
                img_inp_clip = data['image_clip_input']
                txt_inp_clip = data['text_clip_input']
                with torch.no_grad():
                    img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                    txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)
                
                img_feat_vgg = data['image_vgg_feature']                
                txt_feat_trans = data['text_drob_embedding']

                label = data['label'].to(device)

                model.zero_grad()
                
                output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
#                 output = model(img_feat_vgg, txt_feat_trans)
                
                val_loss = criterion(output.squeeze(), label.float())
                acc = torch.abs(output.squeeze() - label.float()).view(-1)
                acc = (1. - acc.sum() / acc.size()[0])
                total_acc_val += acc
                total_loss_val += val_loss.item()
        print("Saving model...")         
        
        torch.save(model.state_dict(), os.path.join(exp_path, "final.pt"))

        val_acc = total_acc_val/len(dataloader_val)
        val_loss = total_loss_val/len(dataloader_val)

        # wandb.log({"val accuracy: ": val_acc, "val loss: ": val_loss})

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        print(f'Epoch {i+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        model.train()
        torch.cuda.empty_cache()
        
    # load the last checkpoint with the best model
#     model.load_state_dict(torch.load(chk_file))
    
    return  model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, i

# ## Testing

# For BCE loss
def test_model(model, criterion):
    model.eval()
    total_acc_test = 0
    total_loss_test = 0
    outputs = []
    test_labels=[]
    with torch.no_grad():
        for data in dataloader_test:
            img_inp_clip = data['image_clip_input']
            txt_inp_clip = data['text_clip_input']
            with torch.no_grad():
                img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)

            img_feat_vgg = data['image_vgg_feature']
            txt_feat_trans = data['text_drob_embedding']            

            label = data['label'].to(device)
            
#             out = model(img_feat_vgg, txt_feat_trans)        

            out = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)        

            outputs += list(out.cpu().data.numpy())
            loss = criterion(out.squeeze(), label.float())
#             print(out.squeeze())
#             print(label.float())
            acc = torch.abs(out.squeeze() - label.float()).view(-1)
    #         print((acc.sum() / acc.size()[0]))
            acc = (1. - acc.sum() / acc.size()[0])
    #         print(acc)
            total_acc_test += acc
            total_loss_test += loss.item()

    acc_test = total_acc_test/len(dataloader_test)
    loss_test = total_loss_test/len(dataloader_test)
    print(f'acc: {acc_test:.4f} loss: {loss_test:.4f}')
    return outputs

# del model
# path = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
# path = os.path.join(exp_path, "final_covpretrain.pt")
# model = MM(output_size)
# model.load_state_dict(torch.load(path))
# model.to(device)
# print(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

######################################## Start training ##################################
output_size = 1 # Binary case
# output_size = 3
exp_name = config["exp_name"] # experiment name for every run 
# pre_trn_ckp = "EMNLP_MCHarm_GLAREAll_COVTrain" # Uncomment for using pre-trained
exp_path = config["exp_path"]  # path where weights and plots are saved
# initialize the experiment path
Path(exp_path).mkdir(parents=True, exist_ok=True)
lr=config["lr"]
criterion = nn.BCELoss() # Binary case
# criterion = nn.CrossEntropyLoss()
# # ------------Fresh training------------
model = MM(output_size)
model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
n_epochs = config["n_epochs"]
# early stopping patience; how long to wait after last time validation loss improved.
patience = config["patience"]

model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, epoc_num = train_model(model, patience, n_epochs, optimizer)
############################################## Plot the training and validation curves ##########################################

results_path = os.path.join(exp_path, 'results')
Path(results_path).mkdir(parents=True, exist_ok=True)
plot_curves(train_acc_list, val_acc_list, train_loss_list, val_loss_list, results_path)

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



