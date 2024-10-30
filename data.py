import numpy
import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2
import PIL
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as tvmodel
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import time
import random
from torchvision import transforms

"""
    之前做csv的时候，就是num,1470的形式，这里没有动，后边还是要还原的
"""


CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
STATIC_DEBUG = False
GL_NUMGRID = 7
GL_NUMBBOX = 2


#%%
class MyDataset(Dataset):
    def __init__(self, dataset_dir, seed=None, mode="train",trans=None):
        """
		最后训练的时候，用的是图片(x)和csv(y)
		:param dataset_dir: 数据所在文件夹   即之前的root_dir
		:param seed: 打乱数据所用的随机数种子
		:param mode: 数据模式，"train", "val", "test"
		:param train_val_ratio: 训练时，训练集:验证集的比例
		:param trans:  数据预处理函数

		TODO:
		1. 读取储存图片路径的.txt文件，并保存在self.img_list中
		2. 读取储存样本标签的.csv文件，并保存在self.label中
		3. 如果mode="train"， 将数据集拆分为训练集和验证集，用self.use_ids来保存对应数据集的样本序号。
			注意，mode="train"和"val"时，必须传入随机数种子，且两者必须相同
		4. 保存传入的数据增广函数
		"""
        if seed is None:
            seed = random.randint(0, 65536)
        random.seed(seed)
        self.dataset_dir = dataset_dir
        self.mode = mode
        img_list_txt = os.path.join(dataset_dir, mode + ".txt")  # 储存图片位置的列表 path
        label_csv = os.path.join(dataset_dir, mode + ".csv")  # 储存标签的数组文件 path
        #  此处根据 mode==trian/test 选取不同的数据
        self.img_list = []  # 用于存储图片名
        # 读取训练/测试集的标签文件
        self.labels = np.loadtxt(label_csv)

        # 读取训练/测试集中的图片名称               标签文件和名称在位置关系上是一一对应的
        with open(img_list_txt, 'r') as f:
            for line in f.readlines():
                self.img_list.append(line.strip())

        self.num=len(self.img_list)
        #num_all_data存储图片总数
        self.save_img_dir="./VOC2007/Augimage/"

        # # 在mode=train或val时， 将数据进行切分
        # # 注意在mode="val"时，传入的随机种子seed要和mode="train"相同
        # self.num_all_data = len(self.img_list)
        # all_ids = list(range(self.num_all_data))
        # num_train = int(train_val_ratio*self.num_all_data)
        # if self.mode == "train":
        #     self.use_ids = all_ids[:num_train]
        # elif self.mode == "val":
        #     self.use_ids = all_ids[num_train:]
        # else:
        #     self.use_ids = all_ids

        # 储存数据增广函数
        self.trans = trans

    def __len__(self):
        """获取数据集数量"""
        return self.num

    def __getitem__(self, item):
        """
		TODO:
		1. 按顺序依次取出第item个训练数据img及其对应的样本标签label
		2. 图像数据要进行预处理，并最终转换为(c, h, w)的维度，同时转换为torch.tensor
		3. 样本标签要按需要转换为指定格式的torch.tensor
		"""
        label =self.labels[item, :]
        label = torch.tensor(label)


        img_name = self.img_list[item]
        image_path=os.path.join(self.save_img_dir,img_name)
        img = Image.open(image_path)
        if self.trans is None:
            trans = transforms.Compose([
                # transforms.Resize((112,112)),
                transforms.ToTensor(),
            ])
        else:
            trans = self.trans
        img = trans(img)  # 图像预处理&数据增广
        # transforms.ToPILImage()(img).show()  # for debug
        # print(label)
        return img, label

# #%%
# # 调试用，依次取出数据看看是否正确
# dataset_dir = "./VOC2007/root"
# dataset = MyDataset(dataset_dir)
# dataloader = DataLoader(dataset, 1)
# # for i in enumerate(dataloader):
# #     print(i)


# #%%
# for img,label in dataset:
#     print(img.shape)
#     print(label.shape)
#     print(label)
#     break
#
#
# #%%
# print(dataset.labels.shape)