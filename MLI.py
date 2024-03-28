# out0417aid.log
import argparse
import glob
from collections import defaultdict
import os
import numpy as np
import random
import torchvision.transforms as transforms
from torch.nn import Parameter
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import datetime
import time
import logging
from datagen import Datatrans
from mmpretrain import get_model
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('-ds', help='specify a dataset', type=str, default='AID_multilabel',
                    choices=['AID_multilabel', 'UCMerced', 'DFC15_multilabel'], required=False)
parser.add_argument('-ep', help='number of epochs', type=int, default=500)
parser.add_argument('-bs', help='batch size', type=int, default=64)
parser.add_argument('-model', help='model name', type=str, default='resnet50',
                    choices=['resnet50', 'resnet101', 'mlp','alexnet'], required=False)
parser.add_argument('-img', help='img name', type=str, default='jpg',
                    choices=['jpg', 'tif', 'png'], required=False)
parser.add_argument("--lr", default=0.01, help="display a square of a given number", type=float)
parser.add_argument('-dir', help='result save path', type=str, default='results/bestpath', required=False)
parser.add_argument('-f')

args = parser.parse_args()
dir_pth = '/Public/ShiwenZeng/Datasets/Target_Kang_Datasets/'
UCM_ML_npy = os.path.join(dir_pth, 'ucm_ml.npy')  # tif
AID_ML_npy = os.path.join(dir_pth, 'aid_ml.npy')  # jpg
DFC_ML_npy = os.path.join(dir_pth, 'dfc15_ml.npy')  # png

ctx = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q = 0.5


def default_loader(path):
    return Image.open(path).convert('RGB')

b = 0

if args.ds == 'DFC15_multilabel':
    clsNum = 8
    main_label = ['impervious', 'water', 'clutter', 'vegetation', 'building', 'tree', 'boat', 'car']
else:
    clsNum = 17
    main_label = ["airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock", "field",
                  "grass", "mobile-home", "pavement", "sand", "sea", "ship", "tanks", "trees", "water"]



def compute_f1(labels, outputs):
    cpu = torch.device("cpu")
    labels = labels.to(cpu).detach().numpy()
    outputs = outputs.to(cpu).detach().numpy()

    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            if outputs[i][j] > Q:
                outputs[i][j] = 1
            else:
                outputs[i][j] = 0

    F1 = []
    for i in range(labels.shape[0]):
        F1.append(f1_score(labels[i], outputs[i]))
    return np.mean(F1)


def evaluate_loss(dataloader, net):
    net.eval()
    val_loss_sum, n, val_f1_sum, val_acc_sum = 0.0, 0, 0.0, 0.0
    with torch.no_grad():
        for bidx, (input, target) in enumerate(dataloader):
            input_img = input.to(ctx)
            target = target.to(ctx)

            output = net(input_img)

            criterion = nn.MultiLabelSoftMarginLoss()
            loss = criterion(output, target)

            f1_score = compute_f1(target, output)
            val_f1_sum += f1_score.item()
            val_loss_sum += loss.item()

            n = n + 1
            print("[batch {}] val_loss={:.5f},f1={:.5f})".format(bidx + 1, loss, f1_score))
    val_loss = val_loss_sum / n
    val_f1 = val_f1_sum / n
    print("[{}-{}]loss:{:.5f}   f1:{:.5f}".format("val", "end", val_loss, val_f1))
    return val_loss, val_f1


def train(net, dataloader):
    train_loss_sum, n, train_f1_sum, train_acc_sum, m = 0.0, 0, 0.0, 0.0, 0
    net.train()
    for bidx, (input, target,_) in enumerate(dataloader):
        input_img = input.to(ctx)
        target = target.to(ctx)
        optimizer.zero_grad()
        loss_fn = nn.MultiLabelSoftMarginLoss()

        output = net(input_img)

        loss = loss_fn(output, target)

        loss.backward()

        optimizer.step()

        train_loss_sum += loss.item()
        f1s = compute_f1(target, output)
        train_f1_sum += f1s.item()

        n = n + 1
        m += target.size()[0]
        print("[batch {}] train_loss={:.5f},f1={:.5f})".format(bidx + 1, loss, f1s))
    train_loss = train_loss_sum / n
    train_f1 = train_f1_sum / n

    print("[{}-{}]loss:{:.5f}   f1:{:.5f}".format("train", "end", train_loss, train_f1))
    return train_loss, train_f1


if __name__ == "__main__":
    now_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
    print(now_time)
    print(args)
    loss_train = []
    f1_train = []
    acc_train = []
    loss_val = []
    f1_val = []
    acc_val = []
    model_ft = get_model('seresnet50_8xb32_in1k', pretrained=True)
    model_ft.head.fc = torch.nn.Linear(in_features=model_ft.head.fc.in_features, out_features=clsNum, bias=True)
    model = model_ft.to(ctx)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomGrayscale(),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    val_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        normalize])
    train_dataGen = Datatrans(dataset=args.ds,
                              phase="train",
                              way=args.dir,
                              transform=train_transform,
                              cls=clsNum
                              )
    val_dataGen = Datatrans(dataset=args.ds,
                            phase="val",
                            way=args.dir,
                            transform=val_transform,
                            cls=clsNum
                            )

    train_loader = torch.utils.data.DataLoader(train_dataGen, batch_size=args.bs, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataGen, batch_size=args.bs, shuffle=False)

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr)
    global Max_map
    Max_map = 0.0
    if not os.path.exists(args.dir + '/pth'):
        os.makedirs(args.dir + '/pth')
    for i in range(args.ep):
        print(f"Epoch {i + 1}-------------------------------")
        if i % 30 == 0 and i != 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
        train_loss, train_f1 = train(model, train_loader)
        val_loss, val_f1 = evaluate_loss(val_loader, model)
        loss_train.append(train_loss)
        f1_train.append(train_f1)
        #         acc_train.append(train_acc)
        loss_val.append(val_loss)
        f1_val.append(val_f1)
        if (val_f1 > Max_map):
            Max_map = val_f1
            torch.save(model, '{}/{}{}_best{}.pth'.format(args.dir + '/pth', args.model,args.ds,now_time))
            print("[epoch {}][save_best_output_params]".format(i + 1))

