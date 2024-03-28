import os.path
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse

import numpy as np
import random
from datagen import Datatrans
from utils import lum_loss
import torch.hub
import datetime
from mmpretrain import get_model

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
    prog='PRODEN demo file.',
    usage='Demo with partial  labels.',
    description='A simple demo file with MNIST dataset.',
    epilog='end',
    add_help=True,
)

parser = argparse.ArgumentParser()
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-3)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-5, choices=[1e-3, 1e-4, 1e-5, 1e-6])
parser.add_argument('-bs', help='batch size', type=int, default=128)
parser.add_argument('-ep', help='number of epochs', type=int, default=2)
parser.add_argument('-ds', help='specify a dataset', type=str, default='AID_multilabel',
                    choices=['AID_multilabel', 'UCMerced', 'DFC15_multilabel'], required=False)
parser.add_argument('-model', help='model name', type=str, default='mvgg16',
                    choices=['mse_resnet50'], required=True)
parser.add_argument('-decaystep', help='learning rate\'s decay step', type=int, default=500)
parser.add_argument('-decayrate', help='learning rate\'s decay rate', type=float, default=1)
parser.add_argument('-nw', help='multi-process data loading', type=int, default=4, required=False)
parser.add_argument('-dir', help='result save path', type=str, default='results/way2')

parser.add_argument('-f')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()


if args.ds == 'DFC15_multilabel':
    clsNum = 8
    imgExt = 'png'
elif args.ds == 'UCMerced':
    clsNum = 17
    imgExt = 'tif'
else:
    clsNum = 17
    imgExt = 'jpg'

def main(train_loader, val_loader, now_time):
    print(args)
    print('loading dataset...')
    model_ft = get_model('seresnet50_8xb32_in1k', pretrained=True)
    model_ft.head.fc = torch.nn.Linear(in_features=model_ft.head.fc.in_features, out_features=clsNum, bias=True)

    net = model_ft.to(device)


    optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=args.lr)
    val_loss_fn = torch.nn.CrossEntropyLoss()
    train_loss = []
    val_lossm = []
    train_acc = []
    val_acc = []
    correct_train, total_train, correct_val, total_val = 0, 0, 0, 0
    train_loss_sum, val_loss_sum, n, m, best_acc1 = 0, 0, 0, 0, 0.0
    for epoch in range(0, args.ep):

        net.train()

        if epoch % 30 == 0 and epoch != 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
        for i, (images, labels, trues, indexes) in enumerate(train_loader):
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            trues = trues.to(device)
            output = net(images)
            _, pred = torch.max(output.data, 1)
            total_train += images.size(0)
            correct_train += (pred == trues).sum().item()
            loss, new_label = lum_loss(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n = n + 1
            for j, k in enumerate(indexes):
                train_loader.dataset.train_final_labels[k, :] = new_label[j, :].detach()
        training_acc = 100 * float(correct_train) / float(total_train)
        training_loss = train_loss_sum / n
        train_loss.append(training_loss)
        train_acc.append(training_acc)

        net.eval()
        with torch.no_grad():
            for images, _, trues, _ in val_loader:
                images = images.to(device)
                trues = trues.to(device)
                output1 = net(images)

                val_loss_sum += val_loss_fn(output1, trues.long()).item()
                _, pred = torch.max(output1.data, 1)
                total_val += images.size(0)
                correct_val += (pred == trues).sum().item()
                m = m + 1
        valing_acc = 100 * float(correct_val) / float(total_val)
        valing_loss_m = val_loss_sum / m

        val_acc.append(valing_acc)
        val_lossm.append(valing_loss_m)
        if valing_acc > best_acc1:
            best_acc1 = valing_acc
            best_acc1_epoch = epoch

            if not os.path.exists(f'{save_dir}/saved_ckpt_model/{model_name}'):
                os.mkdir(f'{save_dir}/saved_ckpt_model/{model_name}')
            torch.save(
                net, f'{save_dir}/saved_ckpt_model/{model_name}/{args.ds}_best{now_time}.pth')

        with open(save_file, 'a') as file:
            file.write(
                "[{}]:Training Loss:{:.4f},Training Acc:{:.4f},Valing Lossm:{:.4f},Valing Acc:{:.4f}\n".format(epoch,
                                                                                                               training_loss,
                                                                                                               training_acc,
                                                                                                               valing_loss_m,
                                                                                                               valing_acc))
    print(f"amulmain:save best model to saved_ckpt_model/{args.model}_epoch_{best_acc1_epoch}_best\n")


if __name__ == '__main__':
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
    test_transform = transforms.Compose([
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
    test_dataGen = Datatrans(dataset=args.ds,
                             phase="test",
                             way=args.dir,
                             transform=test_transform,
                             cls=clsNum
                             )
    train_loader = torch.utils.data.DataLoader(train_dataGen, batch_size=args.bs, num_workers=args.nw, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataGen, batch_size=args.bs, num_workers=args.nw, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataGen, batch_size=1, num_workers=args.nw, shuffle=False)

    now_time = datetime.datetime.now().strftime('%m_%d_%H_%M')

    save_dir = '/Public/ShiwenZeng/PrLL/NUI-LU/' + args.dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, (args.ds + '_' + args.model + '_' + str(now_time) + '.csv'))

    model_name = f"{args.model}_{args.ds}_bsz{args.bs}_ep{args.ep}_dstep{args.decaystep}_drate{args.decayrate}"
    print("---train-val start---")
    main(train_loader, val_loader, now_time)


