import torch
import os

exp_name = "MOMENTA_OG"

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "exp_name": exp_name,
    "exp_path": os.path.join("EMNLP_ModelCkpt", exp_name),
    "output_size": 1,
    "lr": 0.001,
    "batch_size": 64,
    "n_epochs": 1,
    "patience": 25,
}
