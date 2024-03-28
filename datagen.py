import glob
import torch
import random
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from PIL import Image


def RI(y):
    new_y = y.clone()
    n, c = y.shape[0], y.shape[1]
    avgC = 0
    for i in range(n):
        row = new_y[i, :]
        shortuse = []
        one = int(10000)
        for j in range(len(row)):
            if row[j] == 1:
                shortuse.append(j)
        for k in range(len(shortuse)):
            if k == len(shortuse) - 1:
                row[shortuse[k]] = one

            else:
                row[shortuse[k]] = random.randint(0, one)
                one -= row[shortuse[k]]

        for l in range(len(shortuse)):
            row[shortuse[l]] /= 10000
        avgC += torch.sum(row)

        new_y[i] = row
    avgC = avgC / n
    return new_y, avgC

def default_loader(path):
    return Image.open(path).convert('RGB')

class Datatrans:
    def __init__(self, ds, imgtransform=None, phase='train'):
        self.ds = ds
        self.imgtransform = imgtransform
        self.phase = phase

        self.train_datadict = defaultdict()
        self.val_datadict = defaultdict()
        self.test_datadict = defaultdict()
        self.Createidxdict()

    def Createidxdict(self):
        random.seed(42)
        imgdir_pth = '/Public/ShiwenZeng/PrLL/NUI-LU/imagedata/'
        if self.ds == 'AID_multilabel':  # AID_multilabel
            imgExt = 'jpg'
            dataset = 'AID_prime'
            primelabelpath = 'new_aid_primelabel.csv'
            mullabelpath = 'new_aid_multilabel.csv'
        elif self.ds == 'UCMerced':  # UCMerced
            imgExt = 'tif'
            dataset = 'UCM_prime'
            primelabelpath = 'new_ucm_primelabel.csv'
            mullabelpath = 'new_ucm_multilabel.csv'
        elif self.ds == 'DFC15_multilabel':  # DFC15_multilabel
            imgExt = 'png'
            dataset = 'DFC_prime'
            primelabelpath = 'new_dfc_primelabel.csv'
            mullabelpath = 'new_dfc_multilabel.csv'
        imgdatadir = os.path.join(imgdir_pth, dataset)
        imgsceneList = [os.path.join(imgdatadir, x) for x in sorted(os.listdir(imgdatadir)) if
                        os.path.isdir(os.path.join(imgdatadir, x))]
        mainlabel_path = os.path.join('/Public/ShiwenZeng/PrLL/NUI-LU/labeldata/', primelabelpath)
        mullabel_path = os.path.join('/Public/ShiwenZeng/PrLL/NUI-LU/labeldata/', mullabelpath)
        mainlabel = pd.read_csv(mainlabel_path)
        mullable = pd.read_csv(mullabel_path)

        self.train_numImgs = 0
        self.val_numImgs = 0
        self.test_numImgs = 0

        self.train_count = 0
        self.val_count = 0
        self.test_count = 0

        for _, scenePth in enumerate(imgsceneList):
            subdirImgPth = sorted(
                glob.glob(os.path.join(scenePth, '*.' + imgExt)))
            random.shuffle(subdirImgPth)
            train_subdirImgPth = subdirImgPth[:int(0.8 * len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(0.8 * len(subdirImgPth)):]

            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)

            for imgpth in train_subdirImgPth:
                imgname = imgpth.split("230310/")[1]
                mul = []
                main = []
                for i in range(len(mainlabel["image_name"])):
                    if str(mainlabel["image_name"][i]) == imgname:
                        main.append(mainlabel.loc[i][2])
                        for j in range(len(mullable["IMAGE\LABEL"])):

                            if str(mullable["IMAGE\LABEL"][j]) == imgname:
                                mul.append(mullable.loc[j][1:])
                                break
                    continue

                self.train_datadict[self.train_count] = (imgpth, np.array(mul[0]), np.array(main[0]))

                self.train_count += 1
            for imgpth in test_subdirImgPth:
                imgname = imgpth.split("230310/")[1]
                mul = []
                main = []
                # print(imgpth.split("230310/")[1])
                for i in range(len(mainlabel["image_name"])):
                    if str(mainlabel["image_name"][i]) == imgname:
                        main.append(mainlabel.loc[i][2])
                        for j in range(len(mullable["IMAGE\LABEL"])):
                            if str(mullable["IMAGE\LABEL"][j]) == imgname:
                                mul.append(mullable.loc[j][1:])

                                break
                    continue
                self.test_datadict[self.test_count] = (imgpth, np.array(mul[0]), np.array(main[0]))
                self.test_count += 1

        print("total number of classes: {}".format(len(imgsceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        self.trainDataIndex = list(range(self.train_numImgs))

        self.testDataIndex = list(range(self.test_numImgs))

    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
            img, target, true = self.train_img[idx], self.train_final_labels[idx], self.train_label_main[idx]
            self.train_final_labels, self.average_class_label = RI(torch.from_numpy(self.train_label_mul))
        else:
            idx = self.testDataIndex[index]
            img, target, true = self.test_img[idx], self.test_label_main[idx], self.test_label_main[idx]



        img = default_loader(img)
        if self.imgtransform is not None:
            img = self.imgtransform(img)

        return img,target, true, index


    def __len__(self):

        if self.phase == 'train':
            return len(self.trainDataIndex)
        else:
            return len(self.testDataIndex)






