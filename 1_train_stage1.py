import os
import numpy as np
import argparse
import importlib
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_fun import infer
from tool.log import MyLogger
cudnn.enabled = True

def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc

def train_phase(args):
    # 初始化 TensorBoard，用于实时可视化训练过程中的指标
    writer = SummaryWriter(args.log_dir)

    # 加载神经网络
    model = getattr(importlib.import_module(args.network), 'Net')(args.init_gama, n_class=args.n_class)

    # 打印训练参数，方便调试
    # print(vars(args))
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print('----------')
    print("No PDA used")
    print('----------')

    # 定义数据增强和预处理操作
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        transforms.ToTensor()                    # 转换为张量
    ]) 

    # 加载训练数据集，传入数据路径、变换方法和数据集名称
    train_dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=transform_train, dataset=args.dataset)

    # 定义数据加载器
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True
    )

    # 计算训练的总步数，用于学习率调度
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    # 获取模型参数组，分别设置不同的学习率和权重衰减
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},       # 基础层
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},              # 高层特征
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},   # 分类头
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}             # 其他特征
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    # 加载预训练权重
    if args.weights[-7:] == '.params':
        # 如果是 MXNet 格式的权重，进行转换
        assert args.network == "network.resnet38_cls"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    elif args.weights[-4:] == '.pth':
        # 如果是 PyTorch 格式的权重，直接加载
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    else:
        print('random init')

    # 将模型移动到 GPU
    model = model.cuda()

    # 初始化一个平均器，用于记录损失和准确率
    avg_meter = pyutils.AverageMeter('loss', 'avg_ep_EM', 'avg_ep_acc')

    # 初始化计时器，用于记录训练时间
    timer = pyutils.Timer("Session started: ")

    # 开始训练循环
    for ep in range(args.max_epoches):
        model.train()  # 设置模型为训练模式
        args.ep_index = ep  # 记录当前 epoch 索引
        ep_count = 0  # 当前 epoch 的样本总数
        ep_EM = 0     # 当前 epoch 的精确匹配数
        ep_acc = 0    # 当前 epoch 的累计准确率

        # 遍历训练数据集
        for iter, (filename, data, label) in enumerate(train_data_loader):
            img = data  # 获取图像数据
            label = label.cuda(non_blocking=True)  # 将标签移动到 GPU

            # 是否启用PDA
            # if ep > 2:
            #     enable_PDA = 1
            # else:
            #     enable_PDA = 0
            enable_PDA = 0

            # 前向传播，获取输出和特征
            x, feature, y = model(img.cuda(), enable_PDA)
            # 将 y,label 转移到 CPU 并转换为 NumPy
            prob = y.cpu().data.numpy()  
            gt = label.cpu().data.numpy()

            # 计算精确匹配和准确率
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > 0.5)[0]  # 预测为正类的类别
                true_cls = np.where(gt[num] == 1)[0]  # 实际为正类的类别
                if np.array_equal(pass_cls, true_cls):  # 判断是否完全匹配
                    ep_EM += 1
                acc = compute_acc(pass_cls, true_cls)  # 计算准确率
                ep_acc += acc

            avg_ep_EM = round(ep_EM / ep_count, 4)  # 计算平均精确匹配率
            avg_ep_acc = round(ep_acc / ep_count, 4)  # 计算平均准确率

            # 结合 sigmoid 激活和二分类交叉熵损失,计算多标签分类任务中模型输出与目标标签之间的损失
            loss = F.multilabel_soft_margin_loss(x, label)

            # 记录损失和准确率
            avg_meter.add({
                'loss': loss.item(),
                'avg_ep_EM': avg_ep_EM,
                'avg_ep_acc': avg_ep_acc,
            })

            # 优化
            optimizer.zero_grad()  # 清除梯度
            loss.backward()       # 反向传播
            optimizer.step()      # 更新参数
            torch.cuda.empty_cache()  # 清理显存

            # 每 100 步打印一次日志并更新可视化曲线
            if (optimizer.global_step) % 100 == 0 and (optimizer.global_step) != 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Epoch:%2d' % (ep),
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'avg_ep_EM:%.4f' % (avg_meter.get('avg_ep_EM')),
                      'avg_ep_acc:%.4f' % (avg_meter.get('avg_ep_acc')),
                      'lr: %.6f' % (optimizer.param_groups[0]['lr']), 
                      'Fin:%s' % (timer.str_est_finish()),
                      flush=True)
                print('----------')

                # 更新 TensorBoard 曲线
                writer.add_scalar('Loss/train', avg_meter.pop('loss'), optimizer.global_step)
                writer.add_scalar('Accuracy/Exact_Match', avg_meter.pop('avg_ep_EM'), optimizer.global_step)
                writer.add_scalar('Accuracy/Accuracy', avg_meter.pop('avg_ep_acc'), optimizer.global_step)

        # 在训练过程中动态调整 gama 值
        # if model.gama > 0.65:
        #     model.gama = model.gama * 0.98
        # print('Gama of progressive dropout attention is: %.8f' % (model.gama))

    # 保存训练后的模型
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_' + args.dataset + '.pth'))
    writer.close()

def test_phase(args):
    # 动态加载指定的网络模块
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class)
    model = model.cuda()

    # 加载训练阶段保存的权重
    args.weights = os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_' + args.dataset + '.pth')
    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)

    # 设置模型为 eval 模式
    # 禁用 dropout 和 batch normalization 的更新
    model.eval()

    # 对测试集进行评估
    score = infer(model, args.testroot, args.n_class)
    # print(score)
    # 打印整体分数
    print("===== Evaluation Scores =====")
    print("Pixel Accuracy: {:.4f}".format(score['Pixel Accuracy']))
    print("Mean Accuracy: {:.4f}".format(score['Mean Accuracy']))
    print("Frequency Weighted IoU: {:.4f}".format(score['Frequency Weighted IoU']))
    print("Mean IoU: {:.4f}".format(score['Mean IoU']))

    # 打印每个类别的 IoU
    print("\n===== Class IoU =====")
    for class_id, iou in score['Class IoU'].items():
        print("Class {}: {:.4f}".format(class_id, iou))


if __name__ == '__main__':
    logger = MyLogger(log_root="logs/stage1/luad")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--session_name", default="Stage 1", type=str)
    parser.add_argument("--env_name", default="PDA", type=str)
    parser.add_argument("--model_name", default='PDA', type=str)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='datasets/BCSS-WSSS/train/', type=str)
    parser.add_argument("--testroot", default='datasets/BCSS-WSSS/test/', type=str)
    parser.add_argument("--save_folder", default='checkpoints/',  type=str)
    parser.add_argument("--init_gama", default=1, type=float)
    parser.add_argument("--dataset", default='bcss', type=str)
    parser.add_argument("--log_dir", default=logger.log_dir, type=str)
    args = parser.parse_args()

    train_phase(args)
    test_phase(args)
