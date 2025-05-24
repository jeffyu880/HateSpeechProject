import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Custom_Collator
from transformers import AutoTokenizer
from MemeCLIP import create_model
from StudentCLIP import StudentFusionModel
from student_models import VisualStudentViT, TextStudent
from configs import cfg
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import torch.nn.functional as F

def inference():
    device = cfg.device
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    dataset_test = load_dataset(cfg=cfg, split='test')
    collator = Custom_Collator(cfg)
    test_loader = DataLoader(dataset_test, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    # Load teacher
    teacher = create_model(cfg)
    teacher.load_state_dict(torch.load(cfg.checkpoint_file)['state_dict'])
    teacher.to(device).eval()

    # Load student
    visual_student = VisualStudentViT(output_dim=cfg.map_dim).to(device)
    text_student = TextStudent(output_dim=cfg.map_dim).to(device)
    student = StudentFusionModel(cfg, visual_student, text_student).to(device)
    student.load_state_dict(torch.load("student_fusion_model.pt"))
    student.eval()

    teacher_preds = []
    student_preds = []
    labels_all = []
    sample_ids = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch['labels'].to(device)
            labels_all.append(labels)

            img_feats = batch['image_features'].to(device)
            txt_feats = batch['text_features'].to(device)
            image_input = torch.stack([teacher.preprocess(img) for img in batch['raw_images']]).to(device)
            text_tokens = tokenizer(batch['raw_texts'], return_tensors='pt', padding=True, truncation=True).to(device)

            # Teacher
            image_proj = teacher.image_map(img_feats)
            text_proj = teacher.text_map(txt_feats)
            img_adapt = teacher.img_adapter(image_proj)
            txt_adapt = teacher.text_adapter(text_proj)
            img_feat = cfg.ratio * img_adapt + (1 - cfg.ratio) * image_proj
            txt_feat = cfg.ratio * txt_adapt + (1 - cfg.ratio) * text_proj
            fused_feat = F.normalize(img_feat, dim=-1) * F.normalize(txt_feat, dim=-1)
            fused_feat = teacher.pre_output(fused_feat)
            logits_teacher = teacher.classifier(fused_feat)
            pred_teacher = torch.argmax(logits_teacher, dim=1)
            teacher_preds.append(pred_teacher)

            # Student
            img_pred, txt_pred, logits_student = student(image_input, text_tokens['input_ids'], text_tokens['attention_mask'])
            pred_student = torch.argmax(logits_student, dim=1)
            student_preds.append(pred_student)

            sample_ids += batch['idx_meme']

    teacher_preds = torch.cat(teacher_preds).cpu()
    student_preds = torch.cat(student_preds).cpu()
    labels_all = torch.cat(labels_all).cpu()

    # Overlap computation
    same_preds = (teacher_preds == student_preds)
    correct_teacher = (teacher_preds == labels_all)
    correct_student = (student_preds == labels_all)
    both_correct = correct_teacher & correct_student

    # Venn Diagram
    plt.figure(figsize=(6, 6))
    venn2(subsets={
        '10': correct_teacher.sum().item() - both_correct.sum().item(),
        '01': correct_student.sum().item() - both_correct.sum().item(),
        '11': both_correct.sum().item()
    },
    set_labels=('Teacher Correct', 'Student Correct'))
    plt.title("Venn Diagram of Correct Predictions")
    plt.savefig("venn_teacher_student.png")
    plt.show()

    print("Teacher Accuracy:", correct_teacher.float().mean().item())
    print("Student Accuracy:", correct_student.float().mean().item())
    print("Both Correct:", both_correct.float().mean().item())

if __name__ == "__main__":
    inference()
