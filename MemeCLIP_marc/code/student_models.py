# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from transformers import DistilBertModel

class VisualStudent(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.backbone = resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, output_dim)
        # self.fc2 = nn.Linear(output_dim,output_dim)

    def forward(self, x):
        return self.fc(self.backbone(x))

class TextStudent(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.proj = nn.Linear(self.encoder.config.hidden_size, output_dim)
        # self.proj2 = nn.Linear(output_dim,output_dim)
    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.proj(x)

class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.scale = scale
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return self.scale * F.linear(x, weight)

class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, visual_feat, text_feat):
        visual_feat = F.normalize(visual_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        return visual_feat * text_feat

class DistilledClassifierModel(nn.Module):
    def __init__(self, visual_student, text_student, num_classes, feature_dim):
        super().__init__()
        self.visual_student = visual_student
        self.text_student = text_student
        self.fusion = FeatureFusion()
        self.classifier = CosineClassifier(feat_dim=feature_dim, num_classes=num_classes)

    def forward(self, image, input_ids, attention_mask):
        vis = self.visual_student(image)
        txt = self.text_student(input_ids, attention_mask)
        fused = self.fusion(vis, txt)
        logits = self.classifier(fused)
        return logits, fused
