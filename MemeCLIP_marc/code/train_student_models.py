import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Custom_Collator
from configs import cfg
from transformers import AutoTokenizer
from StudentCLIP import StudentFusionModel
from student_models import VisualStudent, TextStudent
from MemeCLIP import create_model
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torchmetrics

def cosine_loss(student, teacher):
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    return 1 - F.cosine_similarity(student, teacher, dim=-1).mean()

def mse_loss(student, teacher):
    return F.mse_loss(student, teacher)

def frobenius_loss(student, teacher):
    return torch.norm(student - teacher, p='fro') / student.size(0)

def evaluate(labels, preds):
    acc = torch.sum(labels == preds).item()
    return acc / len(labels)

def plot_losses(losses, labels, title, save_path):
    for loss, label in zip(losses, labels):
        plt.plot(loss, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def contrastive_loss(student_feats, teacher_feats, temperature=0.07):
    """
    Compute contrastive loss using cosine similarity and cross-entropy.
    Assumes student_feats and teacher_feats are [B, D] tensors.
    """

    logits = student_feats @ teacher_feats.T  # [B, B]
    logits /= temperature

    labels = torch.arange(student_feats.size(0)).long().to(student_feats.device)  # [B]
    loss = F.cross_entropy(logits, labels)
    return loss

def main():
    device = cfg.device
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    os.makedirs("plots", exist_ok=True)

    # Dataset and loaders
    dataset_train = load_dataset(cfg=cfg, split='train')
    dataset_val = load_dataset(cfg=cfg, split='val')
    dataset_test = load_dataset(cfg=cfg, split='test')

    print("Number of training examples:", len(dataset_train))
    print("Number of validation examples:", len(dataset_val))
    print("Number of test examples:", len(dataset_test))

    collator = Custom_Collator(cfg)
    train_loader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(dataset_test, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)

    # Teacher
    teacher = create_model(cfg)
    teacher.load_state_dict(torch.load(cfg.checkpoint_file)['state_dict'])
    teacher.to(device).eval()

    # Students
    visual_student = VisualStudent(output_dim=cfg.map_dim).to(device)
    text_student = TextStudent(output_dim=cfg.map_dim).to(device)
    model = StudentFusionModel(cfg,visual_student, text_student).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    pretrained_weights = teacher.classifier.weight.data.clone()
    model.classifier.weight.data = pretrained_weights
    model.classifier.weight.data.requires_grad = False

    # pretrained_weights_2 = teacher.pre_output[0].weight.data.clone()
    # model.fusion_proj[0].weight.data = pretrained_weights_2

    # Safely extract weights from the first Linear layer of teacher's pre_output
    if isinstance(teacher.pre_output[1], torch.nn.Linear):  # skip Dropout at index 0
        pretrained_weights = teacher.pre_output[1].weight.data.clone()
        pretrained_bias = teacher.pre_output[1].bias.data.clone()

        # Copy weights and bias to student's fusion_proj[0] if it is Linear
        if isinstance(model.fusion_proj[0], torch.nn.Linear):
            model.fusion_proj[0].weight.data.copy_(pretrained_weights)
            model.fusion_proj[0].bias.data.copy_(pretrained_bias)

            # Freeze these weights
            model.fusion_proj[0].weight.requires_grad = False
            model.fusion_proj[0].bias.requires_grad = False
        else:
            print("fusion_proj[0] is not a Linear layer.")
    else:
        print("pre_output[1] is not a Linear layer.")

        # Logs
    loss_cls_train, loss_mse_vis_train, loss_mse_txt_train = [], [], []
    loss_cls_val, loss_mse_vis_val, loss_mse_txt_val = [], [], []
    loss_dist_val = []
    acc_val_log = []
    f1_val_log = []
    auroc_val_log = []

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_losses = {"cls": 0, "mse_vision": 0, "mse_txt": 0}
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{cfg.max_epochs}"):
            img_feats = batch['image_features'].to(device)
            txt_feats = batch['text_features'].to(device)
            with torch.no_grad():
                img_target = teacher.img_adapter(teacher.image_map(img_feats))
                txt_target = teacher.text_adapter(teacher.text_map(txt_feats))
                img_target_u = cfg.ratio * img_target + (1 - cfg.ratio) * teacher.image_map(img_feats)
                img_target = img_target_u / img_target_u.norm(dim=-1, keepdim=True)
                txt_target_u = cfg.ratio * txt_target + (1 - cfg.ratio) * teacher.text_map(txt_feats) 
                txt_target = txt_target_u / txt_target_u.norm(dim=-1, keepdim=True)

            images_raw = batch['raw_images']
            images = torch.stack([teacher.preprocess(img) for img in images_raw]).to(device)

            text_raw = batch['raw_texts']
            tokens = tokenizer(text_raw, return_tensors='pt', padding=True, truncation=True).to(device)

            img_pred, txt_pred, logits = model(images, tokens['input_ids'], tokens['attention_mask'])
            # l1 = 1000*mse_loss(txt_pred, txt_target)
            # l2 = 1000*mse_loss(img_pred, img_target)
            l1 = 0.2*contrastive_loss(student_feats=txt_pred, teacher_feats=txt_target)
            l2 = 0.2*contrastive_loss(student_feats=img_pred, teacher_feats=img_target)
            l_cls = torch.nn.CrossEntropyLoss()(logits, batch['labels'].to(device))
            y_preds = torch.argmax(logits, dim=1)
            total_loss = l1 + l2 + l_cls
            # total_loss = l_cls
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_losses["mse_vision"] += l2.item()
            epoch_losses["mse_txt"] += l1.item()
            epoch_losses["cls"] += l_cls.item()

        # Log train
        loss_mse_txt_train.append(epoch_losses["mse_txt"] / len(train_loader))
        loss_mse_vis_train.append(epoch_losses["mse_vision"] / len(train_loader))
        loss_cls_train.append(epoch_losses["cls"] / len(train_loader))
        print(f"Epoch {epoch+1}: Train MSE Vision Loss = {loss_mse_vis_train[-1]:.4f}, Train MSE Text Loss = {loss_mse_txt_train[-1]:.4f} ,Train CLS Loss = {loss_cls_train[-1]:.4f} ")

        # Validation
        model.eval()
        epoch_losses = {"cls": 0, "mse_vision": 0, "mse_txt": 0}

        all_preds = []
        all_labels = []
        all_logits = []
        with torch.no_grad():
            for batch in val_loader:
                img_feats = batch['image_features'].to(device)
                txt_feats = batch['text_features'].to(device)
                img_target = teacher.img_adapter(teacher.image_map(img_feats))
                txt_target = teacher.text_adapter(teacher.text_map(txt_feats))
                img_target_u = cfg.ratio * img_target + (1 - cfg.ratio) * teacher.image_map(img_feats)
                img_target = img_target_u / img_target_u.norm(dim=-1, keepdim=True)
                txt_target_u = cfg.ratio * txt_target + (1 - cfg.ratio) * teacher.text_map(txt_feats)
                txt_target = txt_target_u / txt_target_u.norm(dim=-1, keepdim=True)

                images_raw = batch['raw_images']
                images = torch.stack([teacher.preprocess(img) for img in images_raw]).to(device)

                text_raw = batch['raw_texts']
                tokens = tokenizer(text_raw, return_tensors='pt', padding=True, truncation=True).to(device)

                img_pred, txt_pred, logits = model(images, tokens['input_ids'], tokens['attention_mask'])
                logit_proxy = torch.sigmoid(logits)
                # l1 = 1000*mse_loss(txt_pred, txt_target)
                # l2 = 1000*mse_loss(img_pred, img_target)
                l1 = 0.2*contrastive_loss(student_feats=txt_pred, teacher_feats=txt_target)
                l2 = 0.2*contrastive_loss(student_feats=img_pred, teacher_feats=img_target)
                l_cls = torch.nn.CrossEntropyLoss()(logits, batch['labels'].to(device))
                y_preds = torch.argmax(logits, dim=1)

                all_preds.append(y_preds)
                all_labels.append(batch['labels'].to(device))
                all_logits.append(logit_proxy)

                
                total_loss = l2  + l1 + l_cls
                # total_loss = l_cls
                
                epoch_losses["cls"] += l_cls.item()
                epoch_losses["mse_txt"] += l1.item()
                epoch_losses["mse_vision"] += l2.item()

        loss_mse_vis_val.append(epoch_losses["mse_vision"] / len(val_loader))
        loss_cls_val.append(epoch_losses["cls"] / len(val_loader))
        loss_mse_txt_val.append(epoch_losses["mse_txt"] / len(val_loader))
        print(f"Epoch {epoch+1}: Val MSE Loss Vision = {loss_mse_vis_val[-1]:.4f}, Val MSE Loss Text = {loss_mse_txt_val[-1]:.4f}, Val CLS Loss = {loss_cls_val[-1]:.4f}")

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)

        acc = model.acc(all_preds, all_labels)
        f1 = model.f1(all_preds, all_labels)
        auroc = model.auroc(all_logits, all_labels)
        
        print(f"Epoch {epoch+1}: Val Accuracy = {acc:.4f}, F1 = {f1:.4f}, AUROC = {auroc:.4f}")
        acc_val_log.append(acc.detach().cpu().item())
        f1_val_log.append(f1.detach().cpu().item())
        auroc_val_log.append(auroc.detach().cpu().item())

        model.acc.reset()
        model.f1.reset()
        model.auroc.reset()

    # Save model and plots
    torch.save(model.state_dict(), "student_fusion_model.pt")
    plot_losses([loss_mse_vis_train, loss_mse_vis_val], ['Train', 'Val'], 'MSE Loss', '/pvc/home/MemeCLIP/code/plots/mseVis_loss.png')
    plot_losses([loss_mse_txt_train, loss_mse_txt_val], ['Train', 'Val'], 'MSE Loss', '/pvc/home/MemeCLIP/code/plots/msetxt_loss.png')
    plot_losses([loss_cls_train, loss_cls_val], ['Train', 'Val'], 'Cross Entropy Loss', '/pvc/home/MemeCLIP/code/plots/cls_loss.png')

    plt.plot(acc_val_log)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("/pvc/home/MemeCLIP/code/plots/val_accuracy.png")
    plt.close()

    plt.plot(f1_val_log)
    plt.title("Validation F1-Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.savefig("/pvc/home/MemeCLIP/code/plots/val_f1.png")
    plt.close()

    plt.plot(auroc_val_log)
    plt.title("Validation AUROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.grid(True)
    plt.savefig("/pvc/home/MemeCLIP/code/plots/val_auroc.png")
    plt.close()

    # TEST LOOP:

    print("########## TESTING LOOP ###########")
    model.acc.reset()
    model.f1.reset()
    model.auroc.reset()

    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            img_feats = batch['image_features'].to(device)
            txt_feats = batch['text_features'].to(device)
            img_target = teacher.img_adapter(teacher.image_map(img_feats))
            txt_target = teacher.text_adapter(teacher.text_map(txt_feats))
            img_target_u = cfg.ratio * img_target + (1 - cfg.ratio) * teacher.image_map(img_feats)
            img_target = img_target_u / img_target_u.norm(dim=-1, keepdim=True)
            txt_target_u = cfg.ratio * txt_target + (1 - cfg.ratio) * teacher.text_map(txt_feats)
            txt_target = txt_target_u / txt_target_u.norm(dim=-1, keepdim=True)

            images_raw = batch['raw_images']
            images = torch.stack([teacher.preprocess(img) for img in images_raw]).to(device)

            text_raw = batch['raw_texts']
            tokens = tokenizer(text_raw, return_tensors='pt', padding=True, truncation=True).to(device)

            img_pred, txt_pred, logits = model(images, tokens['input_ids'], tokens['attention_mask'])
            logit_proxy = torch.sigmoid(logits)
            l1 = mse_loss(txt_pred, txt_target)
            l2 = mse_loss(img_pred, img_target)
            l_cls = torch.nn.CrossEntropyLoss()(logits, batch['labels'].to(device))
            y_preds = torch.argmax(logits, dim=1)

            all_preds.append(y_preds)
            all_labels.append(batch['labels'].to(device))
            all_logits.append(logit_proxy)

            
            total_loss = l2  + l1 + l_cls
            
            epoch_losses["cls"] += l_cls.item()
            epoch_losses["mse_txt"] += l1.item()
            epoch_losses["mse_vision"] += l2.item()

    loss_mse_vis_val.append(epoch_losses["mse_vision"] / len(val_loader))
    loss_cls_val.append(epoch_losses["cls"] / len(val_loader))
    loss_mse_txt_val.append(epoch_losses["mse_txt"] / len(val_loader))
    print(f"TESTING: Test MSE Loss Vision = {loss_mse_vis_val[-1]:.4f}, Test MSE Loss Text = {loss_mse_txt_val[-1]:.4f}, Test CLS Loss = {loss_cls_val[-1]:.4f}")

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    acc = model.acc(all_preds, all_labels)
    f1 = model.f1(all_preds, all_labels)
    auroc = model.auroc(all_logits, all_labels)
    
    print(f"TESTING: Test Accuracy = {acc:.4f}, F1 = {f1:.4f}, AUROC = {auroc:.4f}")



if __name__ == '__main__':
    main()
