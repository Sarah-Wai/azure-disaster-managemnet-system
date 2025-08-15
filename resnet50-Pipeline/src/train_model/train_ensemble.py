import os
import cv2
import random
import argparse
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, models
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from azureml.core import Run

# ---------------- Albumentations wrapper for torchvision datasets ----------------
class AlbumentationsTransform:
    """
    Wraps an Albumentations Compose so that it can be used as a torchvision-compatible
    transform: callable(PIL.Image) -> torch.FloatTensor(C,H,W)
    """
    def __init__(self, augmentations: A.Compose):
        self.augmentations = augmentations

    def __call__(self, img):
        # PIL -> numpy (RGB) -> OpenCV BGR is fine because Normalize is channel-wise
        np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        augmented = self.augmentations(image=np_img)
        return augmented["image"]

# ---------------- Balanced Oversampled Dataset ----------------
class BalancedDataset(Dataset):
    """
    Oversamples classes to the size of the largest class.
    Expects an Albumentations Compose in `transform` (called with keyword: image=...).
    """
    def __init__(self, data_dir, transform=None):
        self.dataset = datasets.ImageFolder(data_dir)
        self.transform = transform

        class_to_samples = {}
        for path, label in self.dataset.samples:
            class_to_samples.setdefault(label, []).append((path, label))

        max_count = max(len(samples) for samples in class_to_samples.values())

        self.samples = []
        for label, samples in class_to_samples.items():
            if len(samples) < max_count:
                oversampled = random.choices(samples, k=max_count)
            else:
                oversampled = random.sample(samples, k=max_count)
            self.samples.extend(oversampled)

        self.class_to_idx = self.dataset.class_to_idx
        self.classes = self.dataset.classes

        # Print class distribution after oversampling
        labels = [label for _, label in self.samples]
        label_counts = Counter(labels)
        print("BalancedDataset class distribution after oversampling:")
        for label, count in sorted(label_counts.items()):
            class_name = self.classes[label]
            print(f"  Class {label} ({class_name}): {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Albumentations expects numpy array; we pass as keyword image=...
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']  # torch.FloatTensor CHW
        return image, label

# ---------------- Logging Setup ----------------
def setup_logging():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO
    )
    return logging.getLogger(__name__)

# ---------------- Evaluation Helpers ----------------
def evaluate_predictions(y_true, y_pred, class_names, logger, split_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    run = Run.get_context()

    acc = accuracy_score(y_true, y_pred)
    run.log(f"{split_name} Accuracy", float(acc))
    logger.info(f"{split_name} Accuracy: {acc:.4f}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    for cname, p, r, f in zip(class_names, precision, recall, f1):
        run.log(f"{split_name} Precision - {cname}", float(p))
        run.log(f"{split_name} Recall - {cname}", float(r))
        run.log(f"{split_name} F1-score - {cname}", float(f))
        logger.info(f"{split_name} {cname} - Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    run.log(f"{split_name} Precision (Macro Avg)", float(p_macro))
    run.log(f"{split_name} Recall (Macro Avg)", float(r_macro))
    run.log(f"{split_name} F1-score (Macro Avg)", float(f_macro))

    run.log(f"{split_name} Precision (Weighted Avg)", float(p_weight))
    run.log(f"{split_name} Recall (Weighted Avg)", float(r_weight))
    run.log(f"{split_name} F1-score (Weighted Avg)", float(f_weight))

    logger.info(f"{split_name} Macro Avg - Precision: {p_macro:.4f}, Recall: {r_macro:.4f}, F1: {f_macro:.4f}")
    logger.info(f"{split_name} Weighted Avg - Precision: {p_weight:.4f}, Recall: {r_weight:.4f}, F1: {f_weight:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{split_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(output_dir, f"{split_name.lower()}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    run.upload_file(name=f"outputs/{split_name.lower()}_confusion_matrix.png", path_or_stream=cm_path)
    try:
        run.log_image(name=f"{split_name} Confusion Matrix", path=cm_path)
    except Exception:
        logger.info("run.log_image not available; uploaded image to outputs instead.")

    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    report_path = os.path.join(output_dir, f"{split_name.lower()}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)
    run.upload_file(name=f"outputs/{split_name.lower()}_classification_report.txt", path_or_stream=report_path)

    npy_path = os.path.join(output_dir, f"{split_name.lower()}_confusion_matrix.npy")
    np.save(npy_path, cm)
    run.upload_file(name=f"outputs/{split_name.lower()}_confusion_matrix.npy", path_or_stream=npy_path)

# ---------------- Ensemble Evaluation ----------------
def ensemble_classification_report(models, loader, class_names, logger, output_dir, device, split_name="Train"):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits_sum = None
            for m in models:
                m.eval()
                m.to(device)
                out = m(imgs)
                logits_sum = out if logits_sum is None else logits_sum + out

            avg_logits = logits_sum / len(models)
            preds = avg_logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    os.makedirs(output_dir, exist_ok=True)

    evaluate_predictions(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        logger=logger,
        split_name=split_name,
        output_dir=output_dir
    )

    pred_df = pd.DataFrame({
        "TrueLabel": [class_names[i] for i in all_labels],
        "PredictedLabel": [class_names[i] for i in all_preds]
    })
    csv_path = os.path.join(output_dir, f"{split_name.lower()}_predictions.csv")
    pred_df.to_csv(csv_path, index=False)
    run = Run.get_context()
    run.upload_file(name=f"outputs/{split_name.lower()}_predictions.csv", path_or_stream=csv_path)

    return np.array(all_preds), np.array(all_labels), csv_path

# ---------------- Train with early stopping and OneCycleLR ----------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, logger, output_dir, model_name):
    run = Run.get_context()
    model.to(device)

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    best_model_wts = None

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step scheduler per batch

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / (total_train if total_train > 0 else 1)
        train_acc = correct_train / (total_train if total_train > 0 else 1)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= (total_val if total_val > 0 else 1)
        val_acc = correct_val / (total_val if total_val > 0 else 1)

        logger.info(
            f"[{model_name}] Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        run.log_row(f"{model_name} Loss", epoch=epoch + 1, Train=float(epoch_loss), Validation=float(val_loss))
        run.log_row(f"{model_name} Accuracy", epoch=epoch + 1, Train=float(train_acc), Validation=float(val_acc))

        run.log(f"{model_name} Train Loss", float(epoch_loss))
        run.log(f"{model_name} Validation Loss", float(val_loss))
        run.log(f"{model_name} Train Accuracy", float(train_acc))
        run.log(f"{model_name} Validation Accuracy", float(val_acc))

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            patience_counter = 0
            logger.info(f"[{model_name}] New best val loss: {best_loss:.4f} - saving weights")
            torch.save(best_model_wts, os.path.join(output_dir, f"{model_name}_best.pth"))
        else:
            patience_counter += 1
            logger.info(f"[{model_name}] No improvement, patience {patience_counter}/{patience_limit}")
            if patience_counter >= patience_limit:
                logger.info(f"[{model_name}] Early stopping triggered")
                break

    if best_model_wts:
        model.load_state_dict(best_model_wts)
    return model

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True, help="Train dataset folder (ImageFolder root)")
    parser.add_argument('--test_data', type=str, required=True, help="Test dataset folder (ImageFolder root)")
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging()
    run = Run.get_context()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Albumentations pipelines ---
    train_augs = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_augs = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # --- Build balanced training dataset and a *separate* balanced validation dataset ---
    full_train_ds_for_train = BalancedDataset(args.train_data, transform=train_augs)
    full_train_ds_for_val = BalancedDataset(args.train_data, transform=val_augs)

    n = len(full_train_ds_for_train)
    train_size = int(0.8 * n)
    indices = torch.randperm(n).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_subset = Subset(full_train_ds_for_train, train_idx)  # with augmentations
    val_subset = Subset(full_train_ds_for_val, val_idx)        # without heavy augs (val_augs)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Test set from a separate folder using ImageFolder + wrapper ---
    test_transform = AlbumentationsTransform(val_augs)
    test_ds = datasets.ImageFolder(args.test_data, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    class_names = full_train_ds_for_train.classes
    logger.info(f"Classes: {class_names}")

    # Compute class weights for weighted loss (use *original* oversampled labels for stability)
    labels = [label for _, label in full_train_ds_for_train.samples]
    label_counts = Counter(labels)
    class_counts = [label_counts[i] for i in range(len(class_names))]
    class_weights = torch.tensor([sum(class_counts) / c for c in class_counts], dtype=torch.float).to(device)
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")

    model_names = ['resnet50', 'efficientnet_b0', 'densenet121']
    trained_models = []

    for name in model_names:
        logger.info(f"Training model: {name}")

        if name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Freeze backbone first
            for p in model.parameters():
                p.requires_grad = False
            # Unfreeze the last block
            for p in model.layer4.parameters():
                p.requires_grad = True
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, len(class_names)))
            for p in model.fc.parameters():
                p.requires_grad = True

        elif name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            for p in model.parameters():
                p.requires_grad = False
            for p in model.features[-1].parameters():
                p.requires_grad = True
            in_features = model.classifier[1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, len(class_names)))
            for p in model.classifier.parameters():
                p.requires_grad = True

        elif name == 'densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            for p in model.parameters():
                p.requires_grad = False
            for p in model.features[-1].parameters():
                p.requires_grad = True
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, len(class_names)))
            for p in model.classifier.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown model name: {name}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=args.epoch,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
        )

        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epoch, logger, args.output_dir, name)
        trained_models.append(model)

    # --- Ensemble evaluation ---
    logger.info("Ensemble evaluation on training (val_subset) set...")
    y_pred_train, y_true_train, train_csv = ensemble_classification_report(
        trained_models, val_loader, class_names, logger, args.output_dir, device, split_name="Train"
    )

    logger.info("Ensemble evaluation on test set...")
    y_pred_test, y_true_test, test_csv = ensemble_classification_report(
        trained_models, test_loader, class_names, logger, args.output_dir, device, split_name="Test"
    )

    overall_train_acc = float(accuracy_score(y_true_train, y_pred_train))
    overall_test_acc = float(accuracy_score(y_true_test, y_pred_test))
    run.log("Overall Train Accuracy", overall_train_acc)
    run.log("Overall Test Accuracy", overall_test_acc)
    logger.info(f"Overall Train Accuracy: {overall_train_acc:.4f}")
    logger.info(f"Overall Test Accuracy: {overall_test_acc:.4f}")

    # Per-class test accuracy logging
    cm_test = confusion_matrix(y_true_test, y_pred_test, labels=list(range(len(class_names))))
    class_acc_test = cm_test.diagonal().astype(float) / np.maximum(cm_test.sum(axis=1).astype(float), 1e-8)
    for cname, acc in zip(class_names, class_acc_test):
        run.log(f"Test Accuracy - {cname}", float(acc))

    # Combine CSVs
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_csv_path = os.path.join(args.output_dir, "combined_predictions.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    run.upload_file(name="outputs/combined_predictions.csv", path_or_stream=combined_csv_path)
    run.log("Combined CSV Path", combined_csv_path)

    logger.info("Training and evaluation complete.")

if __name__ == "__main__":
    main()
