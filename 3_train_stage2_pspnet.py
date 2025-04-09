import logging
import os
import time
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
import math
import random
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pprint import pformat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 可用模型列表
MODEL_CHOICES = {
    'unet': smp.Unet,
    'pspnet': smp.PSPNet,
    'deeplabv3plus': smp.DeepLabV3Plus,
    'fpn': smp.FPN,
    'linknet': smp.Linknet,
    'pan': smp.PAN
}

# 可用编码器列表
ENCODER_CHOICES = [
    'timm-resnest101e',
    'resnet101',
    'resnet152',
    'efficientnet-b5',
    'mobilenet_v2'
]


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
class ValPathologyDataset(Dataset):
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


class TrainPathologyDataset(Dataset):
    def __init__(self, image_dir, mask_dirs, transform=None):
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.image_filenames = sorted(os.listdir(image_dir))
        for d in mask_dirs:
            assert len(os.listdir(d)) == len(self.image_filenames), f"掩膜目录 {d} 文件数量不匹配"
        self.mask_filenames = [sorted(os.listdir(d)) for d in mask_dirs]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = []
        for level, (mask_dir, filenames) in enumerate(zip(self.mask_dirs, self.mask_filenames)):
            mask_path = os.path.join(mask_dir, filenames[idx])
            mask = Image.open(mask_path)
            mask = np.array(mask).astype(np.float32)
            masks.append(mask)

        if self.transform:
            transform_input = {'image': image}
            transform_input.update({f'mask{i}': mask for i, mask in enumerate(masks)})

            augmented = self.transform(**transform_input)

            image = augmented['image']
            masks = [augmented[f'mask{i}'] for i in range(len(masks))]
            
        masks = [mask.long() for mask in masks]

        return image, masks


def get_train_transform(num_masks=3):  # 假设有 3 个 mask，数量可调
    additional_targets = {f'mask{i}': 'mask' for i in range(num_masks)}

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.GaussianBlur(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
        ToTensorV2()
    ], additional_targets=additional_targets)


def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    ])


# ------------- 3. 模型定义 -------------
def load_pspnet(num_classes):
    model = smp.PSPNet(
        encoder_name="timm-resnest101e",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    return model.to(device)


def create_model(model_name, encoder_name, num_classes):
    model_class = MODEL_CHOICES[model_name.lower()]
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
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
    diag = np.diag(confusion_matrix)
    acc = np.sum(diag) / np.sum(confusion_matrix)
    ious = diag / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - diag)
    miou = np.nanmean(ious)
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    fwiou = (freq[freq > 0] * ious[freq > 0]).sum()
    return miou, fwiou, acc, ious


# ------------- 5. 学习率调度 -------------
def poly_lr_scheduler(optimizer, lr, epoch_iter, max_iters, power=0.9):
    """Poly 学习率衰减策略"""
    new_lr = lr * pow((1 - epoch_iter / max_iters), power)
    optimizer.param_groups[0]['lr'] = new_lr
    if len(optimizer.param_groups) != 1:
        for i in range(1, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = new_lr * 10
    return new_lr


# ------------- 6. 损失函数 -------------
class ONSSLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, pseudo_label: torch.Tensor):
        """
        Args:
            pred: 模型输出 [B, C, H, W]
            pseudo_label: 伪标签 [B, H, W]
        Returns:
            加权后的损失值
        """
        B, C, H, W = pred.shape

        # Step 1: 计算逐像素交叉熵损失
        loss_map = F.cross_entropy(pred, pseudo_label, ignore_index=self.ignore_index, reduction='none')  # [B, H, W]

        # Step 2: 计算权重矩阵 W
        # 对损失取负，在 H*W 维度做 softmax
        neg_loss = -loss_map  # [B, H, W]
        neg_loss_flat = neg_loss.view(B, -1)  # [B, H*W]
        sm_neg_loss = F.softmax(neg_loss_flat, dim=1).view(B, H, W)  # [B, H, W]

        # 计算均值并归一化
        mean_sm = sm_neg_loss.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        W = sm_neg_loss / (mean_sm + 1e-8)  # [B, H, W]

        # Step 3: 加权损失
        weighted_loss = loss_map * W.detach()
        total_loss = weighted_loss.mean()

        return total_loss


class STLoss(nn.Module):
    def __init__(self, w_d=30, w_a=60):
        super(STLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)
        # distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss_d = F.smooth_l1_loss(d, t_d)
        # angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss_a = F.smooth_l1_loss(s_angle, t_angle)
        loss = self.w_d * loss_d + self.w_a * loss_a
        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
        if not squared:
            res = res.sqrt()
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res


def SWV(outputs_main, outputs_aux1, outputs_aux2, mask):
    n = outputs_main.shape[0]
    loss_main = F.cross_entropy(
        outputs_main, mask.long(), reduction='none').view(n, -1)
    hard_aux1 = torch.argmax(outputs_aux1, dim=1).view(n, -1)
    hard_aux2 = torch.argmax(outputs_aux2, dim=1).view(n, -1)
    loss_select = 0
    for i in range(n):
        aux1_sample = hard_aux1[i]
        aux2_sample = hard_aux2[i]
        loss_sample = loss_main[i]
        agree_aux = (aux1_sample == aux2_sample)
        disagree_aux = (aux1_sample != aux2_sample)
        loss_select += 2 * torch.sum(loss_sample[agree_aux]) + (1 / 2) * torch.sum(loss_sample[disagree_aux])

    return loss_select / (n * loss_main.shape[1])


# ------------- 7. 训练代码 -------------
def train(model, train_loader, val_loader, args, loggers, run_dir, save_dir):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = ONSSLoss()

    best_miou = 0.0
    num_iter = len(train_loader)
    current_lr = args.learning_rate

    with tensorboard_writer(run_dir) as writer:
        for epoch in range(args.num_epochs):
            model.train()
            running_loss = 0.0

            for i, (image, masks_list) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} (LR: {current_lr:.8f})")
            ):
                image = image.to(device)
                masks_list = [m.to(device) for m in masks_list]
                loss_list = []
                current_lr = poly_lr_scheduler(optimizer, args.learning_rate, epoch * num_iter + i,
                                               num_iter * args.num_epochs)
                optimizer.zero_grad()

                outputs = model(image)
                one = torch.ones((outputs.shape[0],1,224,224)).cuda()
                outputs = torch.cat([outputs,(100 * one * (masks_list[0]==4).unsqueeze(dim = 1))],dim = 1)
                if args.onss:
                    loss_list.append(criterion_2(outputs, masks_list[0]))
                else:
                    loss_list.append(criterion_1(outputs, masks_list[0]))

                # loss_list.append(criterion_1(outputs, masks_list[1]))
                # loss_list.append(criterion_1(outputs, masks_list[2]))

                # loss = loss_list[0] * 0.6 + loss_list[1] * 0.2 + loss_list[2] * 0.2
                loss = loss_list[0]

                # output = model(image)
                # output2 = model(image)
                # output3 = model(image)
                # target = masks_list[0]
                # one = torch.ones((output.shape[0], 1, 224, 224)).cuda()
                # one2 = torch.ones((output2.shape[0], 1, 224, 224)).cuda()
                # one3 = torch.ones((output3.shape[0], 1, 224, 224)).cuda()
                # output = torch.cat([output, (100 * one * (target == 4).unsqueeze(dim=1))], dim=1)
                # output2 = torch.cat([output2, (100 * one2 * (target == 4).unsqueeze(dim=1))], dim=1)
                # output3 = torch.cat([output3, (100 * one3 * (target == 4).unsqueeze(dim=1))], dim=1)
                # loss_v1 = SWV(output, output2, output3, target)
                # loss_st1 = STLoss()(output, output2)
                # loss_st2 = STLoss()(output, output3)
                # loss_st = (loss_st1 + loss_st2) / 2
                # loss = 0.8 * loss_v1 + 0.2 * loss_st

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("LearningRate", current_lr, epoch)

            val_loss, miou, fwiou, acc, ious = evaluate_model(args, model, val_loader, num_classes=args.num_classes,
                                                              compute_loss=True)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Metric/MIoU", miou, epoch)
            writer.add_scalar("Metric/FWIoU", fwiou, epoch)
            writer.add_scalar("Metric/Acc", acc, epoch)

            writer.add_scalar("IoU/TE", ious[0], epoch)
            writer.add_scalar("IoU/NEC", ious[1], epoch)
            writer.add_scalar("IoU/LYM", ious[2], epoch)
            writer.add_scalar("IoU/TAS", ious[3], epoch)

            loggers.info(
                f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, \n"
                f"MIoU = {miou:.4f}, FWIoU = {fwiou:.4f}, Acc = {acc:.4f}, \n"
                f"TE IoU = {ious[0]:.4f}, NEC IoU = {ious[1]:.4f}, LYM IoU = {ious[2]:.4f}, TAS IoU = {ious[3]:.4f}"
            )

            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), os.path.join(save_dir, "best", args.model, ".pth"))
                loggers.info("最优模型已保存")

            torch.save(model.state_dict(), os.path.join(save_dir, "latest", args.model, ".pth"))
            loggers.info("最新模型已保存")


# ------------- 8. 评估代码 -------------
def evaluate_model(args, model, data_loader, model_path=None, num_classes=4, compute_loss=False):
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes))
    running_loss = 0.0

    if args.onss:
        criterion = ONSSLoss(ignore_index=4)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=4)

    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            if compute_loss:
                loss = criterion(outputs, masks)
                running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            preds, masks = preds.cpu().numpy(), masks.cpu().numpy()
            preds[masks == 4] = 4

            confusion_matrix += generate_matrix(masks, preds, num_classes)

    miou, fwiou, acc, ious = compute_metrics(confusion_matrix)
    avg_loss = running_loss / len(data_loader) if compute_loss else None

    return avg_loss, miou, fwiou, acc, ious


# ------------- 9. 主程序 -------------
def main(args):
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(args.log_dir, args.dataset, timestamp)
    save_dir = os.path.join(args.checkpoint, args.dataset, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    loggers = setup_logging(run_dir)

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    if args.Grad_CAM_pp:
        mask_path = os.path.join(args.dataroot, "train_PM_pp")
    else:
        mask_path = os.path.join(args.dataroot, "train_PM")

    train_dataset = TrainPathologyDataset(
        image_dir=os.path.join(args.dataroot, "train"),
        mask_dirs=[
            os.path.join(mask_path, "PM_bn7"),
            os.path.join(mask_path, "PM_b5_2"),
            os.path.join(mask_path, "PM_b4_5")
        ],
        transform=train_transform
    )
    val_dataset = ValPathologyDataset(
        image_dir=os.path.join(args.dataroot, "val/img"),
        mask_dir=os.path.join(args.dataroot, "val/mask"),
        transform=val_transform
    )
    test_dataset = ValPathologyDataset(
        image_dir=os.path.join(args.dataroot, "test/img"),
        mask_dir=os.path.join(args.dataroot, "test/mask"),
        transform=val_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model = load_pspnet(num_classes=args.num_classes)
    model = create_model(args.model, args.encoder, args.num_classes)

    loggers.info("训练开始")
    # loggers.info(f"训练参数: {vars(args)}")
    loggers.info("训练参数:\n" + pformat(vars(args), indent=4))

    train(model, train_loader, val_loader, args, loggers, run_dir, save_dir)

    best_model_path = os.path.join(save_dir, "best", args.model, ".pth")
    latest_model_path = os.path.join(save_dir, "latest", args.model, ".pth")

    test_loss, miou, fwiou, acc, ious = evaluate_model(args, model, test_loader, num_classes=args.num_classes, model_path=best_model_path)
    loggers.info(
        f"最优模型测试结果: MIoU = {miou:.4f}, FWIoU = {fwiou:.4f}, Acc = {acc:.4f}, "
        f"TE IoU = {ious[0]:.4f}, NEC IoU = {ious[1]:.4f}, LYM IoU = {ious[2]:.4f}, TAS IoU = {ious[3]:.4f}"
    )

    test_loss, miou, fwiou, acc, ious = evaluate_model(args, model, test_loader, num_classes=args.num_classes, model_path=latest_model_path)
    loggers.info(
        f"最新模型测试结果: MIoU = {miou:.4f}, FWIoU = {fwiou:.4f}, Acc = {acc:.4f}, "
        f"TE IoU = {ious[0]:.4f}, NEC IoU = {ious[1]:.4f}, LYM IoU = {ious[2]:.4f}, TAS IoU = {ious[3]:.4f}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PSPNet Pathology Segmentation Training")
    parser.add_argument("--onss", type=bool, default=False, help="Whether to use onss")
    parser.add_argument("--Grad_CAM_pp", type=bool, default=False, help="Whether to use Grad CAM++")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of training workers")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of segmentation classes")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--log_dir", type=str, default="./runs", help="Directory to save logs")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage2", help="Directory to save checkpoints")
    # parser.add_argument("--dataroot", default="datasets/BCSS-WSSS", type=str)
    # parser.add_argument("--dataset", default="bcss", type=str)
    parser.add_argument("--dataroot", default="datasets/LUAD-HistoSeg", type=str)
    parser.add_argument("--dataset", default="luad", type=str)
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_CHOICES.keys()), help='选择模型架构')
    parser.add_argument('--encoder', type=str, default='timm-resnest101e', choices=ENCODER_CHOICES, help='选择编码器/主干网络')

    args = parser.parse_args()
    torch.cuda.empty_cache()
    main(args)
