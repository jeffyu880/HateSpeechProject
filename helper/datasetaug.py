import torch
import pandas as pd
import numpy as np
import ast
import cv2
import torch.nn as nn
from torchvision import models
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from sentence_transformers import SentenceTransformer
from PIL import Image

# local file imports
from .simpletokenizer import SimpleTokenizer
from .vggmodel import customVGG


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####################### HELPER FUNCTIONS FOR EVALUATING VGG #########################

# VGG16 for extracting image encodings.

# Initialize the model
model_vgg_pretrained = models.vgg16(pretrained=True)
model_vgg = customVGG(model_vgg_pretrained)

# Change the device to GPU
model_vgg = model_vgg.to(device)
# For cropped ROI proposals
# Transform the image, so it becomes readable with the model
transform_vgg_BB = Compose([
  ToPILImage(),
#   transforms.CenterCrop(512),
  Resize((448,448)),
  ToTensor()                              
])

# Iterate each image
def get_image_vgg_BB(l, t, r, b, in_im, device : str, model_vgg : torch.nn): 
#     left, top, right, bottom and input image
    img = cv2.imread(in_im)
    h, w, _ = img.shape
    # crop
    x1 = int(np.floor(l*w))
    x2 = int(np.floor(r*w))
    y1 = int(np.floor(b*h))
    y2 = int(np.floor(t*h))
    crop_img = img[y1:y2, x1:x2]    
    
    # Transform the cropped image
    img = transform_vgg_BB(crop_img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    # We only extract features, so we don't need gradient
    with torch.no_grad():
        # Extract the feature from the image
        feature = model_vgg(img).squeeze()
    # Convert to NumPy Array, Reshape it, and save it to features variable
    return feature

# Transform the image, so it becomes readable with the model
transform_vgg_center = Compose([
  ToPILImage(),
  CenterCrop(512),
  Resize(448),
  ToTensor()                              
])

# Iterate each image
def get_image_vgg_center(in_im, device : str, model_vgg : torch.nn):
    # Set the image path
    path = in_im
    # Read the file
    img = cv2.imread(path)
    # Transform the image
    img = transform_vgg_center(img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    # We only extract features, so we don't need gradient
    with torch.no_grad():
        # Extract the feature from the image
        feature = model_vgg(img).squeeze()
    
    return feature

########################## END HELPER FUNCTIONS FOR VGG #############################


################################ SENTENCE TRANSFORMER ################################
# ### Sentence embedding
# https://github.com/UKPLab/sentence-transformers

model_sent_trans = SentenceTransformer('paraphrase-distilroberta-base-v1')

################################ END SENTENCE TRANSFORMER ################################


class HarmemeMemesDatasetAug2(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path : str,
        img_dir : str,
        train_ROI_path : str,
        val_ROI_path : str,
        test_ROI_path : str,
        train_ENT_path : str,
        val_ENT_path : str,
        test_ENT_path : str,
        input_resolution : int,
        context_length : int,
        clip_model : nn.Module,
        split_flag=None,
        balance=False,
        dev_limit=None,
        random_state=0,
    ):
        self.input_resolution = input_resolution,
        self.context_length = context_length,
        self.samples_frame = pd.read_json(
            data_path, lines=True
        )
        self.samples_frame = self.samples_frame.reset_index(
            drop=True
        )
        self.samples_frame.image = self.samples_frame.apply(
            lambda row: (img_dir + '/' + row.image), axis=1
        )
        if split_flag=='train':
            self.ROI_samples = train_ROI_path
            self.ENT_samples = train_ENT_path
        elif split_flag=='val':
            self.ROI_samples = val_ROI_path
            self.ENT_samples = val_ENT_path
        else:
            self.ROI_samples = test_ROI_path
            self.ENT_samples = test_ENT_path
        
        self.tokenizer = SimpleTokenizer()
        self.preprocess = Compose([
            Resize(self.input_resolution, interpolation=Image.BICUBIC),
            CenterCrop(self.input_resolution),
            ToTensor()
        ])
        self.clip_model = clip_model

    def __len__(self):
        """This method is called when you do len(instance) 
        for an instance of this class.
        """
        return len(self.samples_frame)

    # Get the image features for a single image input
    def process_image_clip(self, in_img):
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
        
        image = self.preprocess(Image.open(in_img).convert("RGB"))
        
        image_input = torch.tensor(np.stack(image)).cuda()
        image_input -= image_mean[:, None, None]
        image_input /= image_std[:, None, None]
        return image_input

    # Get the text features for a single text input
    def process_text_clip(self, in_text):    
        text_token = self.tokenizer.encode(in_text)
        text_input = torch.zeros(self.clip_model.context_length, dtype=torch.long)
        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        tokens = [sot_token] + text_token[:75] + [eot_token]
        text_input[:len(tokens)] = torch.tensor(tokens)
        text_input = text_input.cuda()
        return text_input
    
    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.samples_frame.loc[idx, "id"]
        img_file_name = self.samples_frame.loc[idx, "image"]
        
        image_clip_input = self.process_image_clip(self.samples_frame.loc[idx, "image"])
# --------------------------------------------------------------------------------------        
#         Pre-extracted features
        # FROM SAVED FILE
        image_vgg_feature = self.ROI_samples[idx]        
# --------------------------------------------------------------------------------------
# # On-demand computation
#         BB_info = self.samples_frame.loc[idx, "bbdict"]
#         roi_vgg_feat_list = []
#         if BB_info:
#             total_BB = len(BB_info)
#             if total_BB>4:
#                 BB_info_final = BB_info[:4]
#             else:
#                 BB_info_final = BB_info
# #             Have to get VGG reps for each cropped BB and get the mean             
#             for item in BB_info_final:
# #                 Get the top left (left,top) and bottom right (right,bottom) values of the coordinates
# #                 top left and bottom right value extraction                
#                 left   = item['Vertices'][3][0]
#                 top    = item['Vertices'][3][1]
#                 right  = item['Vertices'][1][0]
#                 bottom = item['Vertices'][1][1]
#                 get_image_vgg_center(img_file_name)
#                 roi_vgg_feat = get_image_vgg_BB(left, top, right, bottom, img_file_name)
#                 roi_vgg_feat_list.append(roi_vgg_feat)
# #             print(np.shape(roi_vgg_feat_list))
# #             print(torch.cat(roi_vgg_feat_list, dim=0))
# #             print(np.mean(np.array(roi_vgg_feat_list), axis=0))
#             image_vgg_feature = torch.mean(torch.vstack(roi_vgg_feat_list), axis=0)
# #             print(image_vgg_feature.shape)
#         else:
#             image_vgg_feature = torch.tensor(get_image_vgg_center(img_file_name))
# --------------------------------------------------------------------------------------
        # FROM SAVED FILE
        text_clip_input = self.process_text_clip(self.samples_frame.loc[idx, "text"])
#         -------------------------------------------------------------------------------
#         Process entities
#        # FROM SAVED FILE
        text_drob_feature = self.ENT_samples[idx]
#         -------------------------------------------------------------------------------
#         Get the mean representation for the set of entities ""on-demand
        # cur_ent_rep_list = []
        # cur_ent_list = self.samples_frame.loc[idx, "ent"]
        
        # if len(cur_ent_list):
        #     for item in cur_ent_list:
        #         cur_ent_rep = torch.tensor(model_sent_trans.encode(item)).to(device)
        #         cur_ent_rep_list.append(cur_ent_rep)
        #     text_drob_feature = torch.mean(torch.vstack(cur_ent_rep_list), axis=0)
        # else:
        #     text_drob_feature = torch.tensor(model_sent_trans.encode(self.samples_frame.loc[idx, "text"])).to(device)
#         -------------------------------------------------------------------------------

        if "labels" in self.samples_frame.columns:
            raw_labels = self.samples_frame.loc[idx, "labels"]

            # If the labels are stored as strings, convert them to actual lists
            if isinstance(raw_labels, str):
                labels_list = ast.literal_eval(raw_labels)
            else:
                labels_list = raw_labels  # Already a list

#             Uncoment below for binary index creation
            if labels_list[0] == "not harmful":
                lab = 0
            else:
                lab = 1
            label = torch.tensor(lab).long()  

#             Uncomment below for one hot encoding
#             y = torch.tensor(lab).to(device)
#             label = F.one_hot(y, num_classes=2)  

# #             Multiclass setting - harmfulness
            # if self.samples_frame.loc[idx, "labels"][0]=="not harmful":
            #     lab=0
            # elif self.samples_frame.loc[idx, "labels"][0]=="somewhat harmful":
            #     lab=1  
            # else:
            #     lab=2
            # label = torch.tensor(lab).to(device)  

            
            sample = {
                "id": img_id, 
                "image_clip_input": image_clip_input,
                "image_vgg_feature": image_vgg_feature,
                "text_clip_input": text_clip_input,
                "text_drob_embedding": text_drob_feature,
                "label": label
            }
        else:
            sample = {
                "id": img_id, 
                "image_clip_input": image_clip_input,
                "image_vgg_feature": image_vgg_feature,
                "text_clip_input": text_clip_input,
                "text_drob_embedding": text_drob_feature
            }

        return sample