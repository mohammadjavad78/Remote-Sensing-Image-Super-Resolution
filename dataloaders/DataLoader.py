import os
import pandas as pd
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2 as cv
from torch import nn as nn, optim as optim
from torchvision import transforms
from torch.utils.data import Dataset


def get_filepaths(directory, target_transform, rand, justin, augh):
    file_paths1 = []
    file_paths2 = []
    labels1 = []
    labels2 = []
    labelsdict = {}
    for root, _, files in os.walk(directory):
        if (len(files) == 100):
            random.seed(rand)
            random.shuffle(files)
            for filename in files[:40]:
                labelsdict[root.split(
                    '/')[-1]] = labelsdict.get(root.split('/')[-1], len(list(labelsdict.keys())))
                filepath = os.path.join(root, filename)
                kk = cv.imread(filepath)
                kk = target_transform(kk)
                file_paths1.append(kk)
                if (augh):
                    file_paths1.append(TF.vflip(kk))
                    file_paths1.append(TF.rotate(kk, 90))
                    file_paths1.append(TF.rotate(kk, 180))
                    file_paths1.append(TF.rotate(kk, 270))
                    labels1.append(labelsdict[root.split('/')[-1]])
                    labels1.append(labelsdict[root.split('/')[-1]])
                    labels1.append(labelsdict[root.split('/')[-1]])
                    labels1.append(labelsdict[root.split('/')[-1]])
                labels1.append(labelsdict[root.split('/')[-1]])
            if (justin == "All" or justin in root):
                for filename in files[40:46]:
                    labelsdict[root.split(
                        '/')[-1]] = labelsdict.get(root.split('/')[-1], len(list(labelsdict.keys())))
                    filepath = os.path.join(root, filename)
                    kk = cv.imread(filepath)
                    kk = target_transform(kk)
                    file_paths2.append(kk)
                    if (augh):
                        file_paths2.append(TF.vflip(kk))
                        file_paths2.append(TF.rotate(kk, 90))
                        file_paths2.append(TF.rotate(kk, 180))
                        file_paths2.append(TF.rotate(kk, 270))
                        labels2.append(labelsdict[root.split('/')[-1]])
                        labels2.append(labelsdict[root.split('/')[-1]])
                        labels2.append(labelsdict[root.split('/')[-1]])
                        labels2.append(labelsdict[root.split('/')[-1]])
                    labels2.append(labelsdict[root.split('/')[-1]])
        elif (len(files) == 800):
            random.seed(rand)
            random.shuffle(files)
            for filename in files[:100]:
                labelsdict[root.split(
                    '/')[-1]] = labelsdict.get(root.split('/')[-1], len(list(labelsdict.keys())))
                filepath = os.path.join(root, filename)
                kk = cv.imread(filepath)
                kk = target_transform(kk)
                file_paths1.append(kk)
                if (augh):
                    file_paths1.append(TF.vflip(kk))
                    file_paths1.append(TF.rotate(kk, 90))
                    file_paths1.append(TF.rotate(kk, 180))
                    file_paths1.append(TF.rotate(kk, 270))
                    labels1.append(labelsdict[root.split('/')[-1]])
                    labels1.append(labelsdict[root.split('/')[-1]])
                    labels1.append(labelsdict[root.split('/')[-1]])
                    labels1.append(labelsdict[root.split('/')[-1]])
                labels1.append(labelsdict[root.split('/')[-1]])
            if (justin == "All" or justin in root):
                for filename in files[101:106]:
                    labelsdict[root.split(
                        '/')[-1]] = labelsdict.get(root.split('/')[-1], len(list(labelsdict.keys())))
                    filepath = os.path.join(root, filename)
                    kk = cv.imread(filepath)
                    kk = target_transform(kk)
                    file_paths2.append(kk)
                    if (augh):
                        file_paths2.append(TF.vflip(kk))
                        file_paths2.append(TF.rotate(kk, 90))
                        file_paths2.append(TF.rotate(kk, 180))
                        file_paths2.append(TF.rotate(kk, 270))
                        labels2.append(labelsdict[root.split('/')[-1]])
                        labels2.append(labelsdict[root.split('/')[-1]])
                        labels2.append(labelsdict[root.split('/')[-1]])
                        labels2.append(labelsdict[root.split('/')[-1]])
                    labels2.append(labelsdict[root.split('/')[-1]])
    return file_paths1, file_paths2, labels1, labels2


class MyDataset(Dataset):
    def __init__(self, train=True, test=False, dir='./drive/MyDrive/Deep_project/datasets/', rand=5, output_size=96, justin="All", augh=True):
        self.dir = dir
        self.train_path_list = []
        self.train_path_list_label = []
        self.train = train
        self.test = test
        self.rand = rand
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((output_size, output_size)),
            transforms.ToTensor()
        ])
        self.target_transform = target_transform
        self.justin = justin

        super(MyDataset, self).__init__()
        # print(os.listdir(self.dir))

        # Run the above function and store its results in a variable.
        train_path_list1, train_path_list2, lab1, lab2 = get_filepaths(
            self.dir, self.target_transform, self.rand, self.justin, augh)

        # print(lab1,lab2)

        if (self.test == True):
            del train_path_list1
            del lab1
            self.imgs = pd.DataFrame(columns=['Address'])
            self.imgslab = pd.DataFrame(columns=['lab'])
            for i in range(len(train_path_list2)):
                # print(train_path_list2.pop()) # adding a row
                self.imgs.loc[self.imgs.size] = [
                    train_path_list2.pop()]  # adding a row
            for i in range(len(lab2)):
                self.imgslab.loc[self.imgslab.size] = [
                    lab2.pop()]  # adding a row
            print(self.imgs.size)
            print(self.imgslab.size)
            del train_path_list2
            del lab2
            # self.imgs=pd.DataFrame.from_dict({'Address':testt['Address']})
            # del testt
        else:
            del train_path_list2
            del lab2
            self.imgs = pd.DataFrame(columns=['Address'])
            self.imgslab = pd.DataFrame(columns=['lab'])
            for i in range(len(train_path_list1)):
                self.imgs.loc[self.imgs.size] = [
                    train_path_list1.pop()]  # adding a row
            for i in range(len(lab1)):
                self.imgslab.loc[self.imgslab.size] = [
                    lab1.pop()]  # adding a row
            print(self.imgs.size)
            print(self.imgslab.size)
            del train_path_list1
            del lab1
            # self.imgs=pd.DataFrame.from_dict({'Address':trainn['Address']})
            # del trainn

        # print(trainn)

        self.resize_hr1 = transforms.Resize(
            (output_size//4, output_size//4), interpolation=transforms.InterpolationMode.BICUBIC)
        self.resize_hr2 = transforms.Resize(
            (output_size//2, output_size//2), interpolation=transforms.InterpolationMode.BICUBIC)

        # self.imgs['img'] = self.imgs['Address'].apply(lambda x: cv.imread(x))

    def __getitem__(self, index):
        img = self.imgs.iloc[index].values[0]
        lab = self.imgslab.iloc[index].values[0]

        resized_img1 = self.resize_hr1(img)
        resized_img2 = self.resize_hr2(img)
        return resized_img1, resized_img2, img, lab

    def __len__(self):
        return len(self.imgs)
