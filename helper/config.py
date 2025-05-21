import torch
import os

exp_name = "MOMENTA_OG"
exp_path = os.path.join("EMNLP_ModelCkpt", exp_name)
weights_path = os.path.join(exp_path, "final.pt")

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "exp_name": exp_name,
    "exp_path": exp_path,
    "output_size": 1,   # binary classification
    "lr": 0.001,
    "batch_size": 64,
    "n_epochs": 25,
    "patience": 25,     # how many epochs before early stopping 
    "train" : False,       # to train the model, if false, test
    "load_weights" : True,   # use pretrained weights to evaluate 
    "weights_path" : weights_path       # path to use to load in weights of model
}
