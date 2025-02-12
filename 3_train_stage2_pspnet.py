import logging
import os
import time
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------- 1. 日志管理 -------------
def setup_logging(log_dir):
    """配置日志记录器，使终端和日志文件保持同步"""
    os.makedirs(log_dir, exist_ok=True)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # 设置全局日志记录器（根 logger）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 日志文件 handler
    log_file = os.path.join(log_dir, "train.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, mode='a')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 终端 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger


@contextmanager
def tensorboard_writer(log_dir):
    """管理 TensorBoard 的 SummaryWriter"""
    writer = SummaryWriter(log_dir)
    try:
        yield writer
    finally:
        writer.close()


# ------------- 2. 数据集定义 -------------
class PathologyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.float32)

        if self.transform:
            image = self.transform(image)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


# ------------- 3. 模型定义 -------------
def load_pspnet(num_classes):
    model = smp.PSPNet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        classes=num_classes,
        activation=None
    )
    return model.to(device)


# ------------- 4. 计算性能指标 -------------
def generate_matrix(gt_image, pre_image, num_class):
    mask = (gt_image >= 0) & (gt_image < num_class)
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=(num_class) ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix


def compute_metrics(confusion_matrix):
    acc = np.diag(confusion_matrix)[0:5].sum() / confusion_matrix.sum()
    ious = np.diag(confusion_matrix)[0:4] / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix,
                                                                                       axis=0) - np.diag(
        confusion_matrix))[0:4]
    miou = np.nanmean(ious)
    freq = np.sum(confusion_matrix, axis=1)[0:4] / np.sum(confusion_matrix)
    fwiou = (freq[freq > 0] * ious[freq > 0]).sum()
    return miou, fwiou, acc, ious


# ------------- 5. 训练代码 -------------
def poly_lr_scheduler(optimizer, init_lr, epoch, max_epochs, power=0.9):
    """Poly 学习率衰减策略"""
    new_lr = init_lr * (1 - epoch / max_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def train(model, train_loader, val_loader, args, loggers, run_dir):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_miou = 0.0

    with tensorboard_writer(run_dir) as writer:
        for epoch in range(args.num_epochs):
            model.train()
            running_loss = 0.0

            # 更新学习率
            current_lr = poly_lr_scheduler(optimizer, args.learning_rate, epoch, args.num_epochs)

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} (LR: {current_lr:.6f})"):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                one = torch.ones((outputs.shape[0], 1, 224, 224), device=device)
                outputs = torch.cat([outputs, (100 * one * (masks == 4).unsqueeze(dim=1))], dim=1)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("LearningRate", current_lr, epoch)

            val_loss, miou, fwiou, acc, ious = evaluate_model(model, val_loader, criterion, compute_loss=True)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Metric/MIoU", miou, epoch)
            writer.add_scalar("Metric/FWIoU", fwiou, epoch)
            writer.add_scalar("Metric/Acc", acc, epoch)

            # 记录 IoU
            writer.add_scalar("IoU/TE", ious[0], epoch)
            writer.add_scalar("IoU/NEC", ious[1], epoch)
            writer.add_scalar("IoU/LYM", ious[2], epoch)
            writer.add_scalar("IoU/TAS", ious[3], epoch)

            loggers.info(
                f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, \n"
                f"MIoU = {miou:.4f}, FWIoU = {fwiou:.4f}, Acc = {acc:.4f}, \n"
                f"TE IoU = {ious[0]:.4f}, NEC IoU = {ious[1]:.4f}, LYM IoU = {ious[2]:.4f}, TAS IoU = {ious[3]:.4f}, \n"
                f"LR = {current_lr:.6f}"
            )

            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), os.path.join(run_dir, "best_pspnet.pth"))
                loggers.info(f"Epoch {epoch + 1}: 最优模型已保存")

            torch.save(model.state_dict(), os.path.join(run_dir, "latest_pspnet.pth"))
            loggers.info("最新模型已保存")


# ------------- 6. 评估代码 -------------
def evaluate_model(model, data_loader, criterion=None, model_path=None, num_classes=5, compute_loss=False):
    """通用评估函数，可用于验证集和测试集。

    参数：
    - model: 需要评估的模型
    - data_loader: 数据集加载器
    - criterion: 损失函数（仅用于验证）
    - model_path: 若提供路径，则加载该模型
    - num_classes: 类别数
    - compute_loss: 是否计算损失（仅在验证时计算）
    """
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes))
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            one = torch.ones((outputs.shape[0], 1, 224, 224), device=device)
            outputs = torch.cat([outputs, (100 * one * (masks == 4).unsqueeze(dim=1))], dim=1)

            if compute_loss and criterion:
                loss = criterion(outputs, masks)
                running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            preds, masks = preds.cpu().numpy(), masks.cpu().numpy()
            preds[masks == 4] = 4  # 确保忽略类正确标记

            confusion_matrix += generate_matrix(masks, preds, num_classes)

    miou, fwiou, acc, ious = compute_metrics(confusion_matrix)
    avg_loss = running_loss / len(data_loader) if compute_loss else None

    return avg_loss, miou, fwiou, acc, ious


# ------------- 7. 主程序 -------------
def main(args):
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    loggers = setup_logging(run_dir)

    transform = transforms.Compose([
        transforms.ToPILImage(),  # 需要转化为 PIL 图像才能应用其他增强
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(30),  # 随机旋转，最大旋转角度30度
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪，比例在0.8到1.0之间
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色变化
        transforms.RandomAffine(30, shear=10),  # 随机仿射变换
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])  # 归一化
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    ])

    train_dataset = PathologyDataset("datasets/LUAD-HistoSeg/train", "datasets/LUAD-HistoSeg/train_PM/PM_bn7",
                                     transform)
    val_dataset = PathologyDataset("datasets/LUAD-HistoSeg/val/img", "datasets/LUAD-HistoSeg/val/mask", transform_val)
    test_dataset = PathologyDataset("datasets/LUAD-HistoSeg/test/img", "datasets/LUAD-HistoSeg/test/mask", transform_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_pspnet(num_classes=args.num_classes)

    loggers.info("训练开始")
    loggers.info(f"训练参数: {vars(args)}")  # 记录所有参数

    train(model, train_loader, val_loader, args, loggers, run_dir)

    # 保存最优模型和最新模型到新的运行目录
    best_model_path = os.path.join(run_dir, "best_pspnet.pth")
    latest_model_path = os.path.join(run_dir, "latest_pspnet.pth")

    # 测试最优模型
    test_loss, miou, fwiou, acc, ious = evaluate_model(model, test_loader, model_path=best_model_path)
    loggers.info(
        f"最优模型测试结果: MIoU = {miou:.4f}, FWIoU = {fwiou:.4f}, Acc = {acc:.4f}, "
        f"TE IoU = {ious[0]:.4f}, NEC IoU = {ious[1]:.4f}, LYM IoU = {ious[2]:.4f}, TAS IoU = {ious[3]:.4f}"
    )

    # 测试最新模型
    test_loss, miou, fwiou, acc, ious = evaluate_model(model, test_loader, model_path=latest_model_path)
    loggers.info(
        f"最新模型测试结果: MIoU = {miou:.4f}, FWIoU = {fwiou:.4f}, Acc = {acc:.4f}, "
        f"TE IoU = {ious[0]:.4f}, NEC IoU = {ious[1]:.4f}, LYM IoU = {ious[2]:.4f}, TAS IoU = {ious[3]:.4f}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PSPNet Pathology Segmentation Training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of training workers")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of segmentation classes")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--log_dir", type=str, default="./runs", help="Directory to save logs")
    parser.add_argument("--save_best_model", action="store_true", help="Save only the best model")
    parser.add_argument("--save_latest_model", action="store_true", help="Save the latest model after each epoch")

    args = parser.parse_args()
    main(args)
