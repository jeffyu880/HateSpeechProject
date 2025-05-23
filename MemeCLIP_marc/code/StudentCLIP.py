import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CosineClassifier
from clip import clip
from configs import cfg
from student_models import VisualStudent, TextStudent
import torchmetrics

class StudentFusionModel(nn.Module):
    def __init__(self, cfg, visual_student, text_student):
        super().__init__()
        self.cfg = cfg
        self.visual_student = visual_student
        self.text_student = text_student

                # Freeze student backbones
        for param in self.visual_student.backbone.parameters():
            param.requires_grad = False

        for param in self.text_student.encoder.parameters():
            param.requires_grad = False

        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes = self.cfg.num_classes)
        self.auroc = torchmetrics.AUROC(task='multiclass', num_classes = self.cfg.num_classes)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes = self.cfg.num_classes, average='macro')

        self.fusion_proj = nn.Sequential(
            nn.Linear(cfg.map_dim, cfg.map_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.drop_probs[2]),
        )

        self.classifier = CosineClassifier(
            feat_dim=1024,
            num_classes=cfg.num_classes,
            dtype=torch.float32,
            scale=cfg.scale
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        # Mimic MemeCLIP prompt initialization
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in self.cfg.class_names]
        prompts = clip.tokenize(prompts, context_length=77).to(self.cfg.device)

        # Load CLIP model just to get projection weights
        model, _ = clip.load(self.cfg.clip_variant, device=self.cfg.device, jit=False)
        model.eval()
        with torch.no_grad():
            text_features = model.encode_text(prompts)  # [num_classes, 768]
            text_features = F.normalize(text_features, dim=-1)
            # Project to 1024-dim using vision tower's projection matrix
            if model.visual.proj is not None:
                text_features = text_features @ model.visual.proj.t()
                text_features = F.normalize(text_features, dim=-1)
            else:
                raise RuntimeError("model.visual.proj is None. Cannot project text features to match classifier input dimension.")
        self.classifier.apply_weight(text_features)

    def safe_normalize(self, x, dim=-1, eps=1e-6):
        return x / (x.norm(dim=dim, keepdim=True).clamp(min=eps))


    def forward(self, image, text_input_ids, text_attention_mask):
        img_feat_u = self.visual_student(image)
        txt_feat_u = self.text_student(text_input_ids, text_attention_mask)

        # Fuse features with element-wise multiplication as in MemeCLIP
        img_feat = img_feat_u / img_feat_u.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat_u / txt_feat_u.norm(dim=-1, keepdim=True)
        fused = torch.mul(img_feat ,txt_feat)
        fused = self.fusion_proj(fused)
        logits = self.classifier(fused.to(self.classifier.weight.dtype)).squeeze(dim = 1)
        
        return img_feat, txt_feat , logits
