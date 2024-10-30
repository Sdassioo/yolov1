""""
xml-txt-csv
xml-labeltxt→augment→txt-cb2-csv
"""

#%%
import PIL.Image
import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2
from PIL.Image import Image
from torch.utils.data import Dataset
import torchvision.models as tvmodel
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import time
import random
from torchvision import transforms

# %%
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
STATIC_DEBUG = False
GL_NUMGRID = 7
GL_NUMBBOX = 2



# %%
def convert(size, box):
    """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
	并进行归一化"""
    img_wid = 1. / size[0]
    img_hei = 1. / size[1]
    x = (box[0] + box[1]) / 2.0  ##横坐标中点
    y = (box[2] + box[3]) / 2.0  ##纵坐标中点
    w = box[1] - box[0]  ##宽
    h = box[3] - box[2]  ##高
    x = x * img_wid
    w = w * img_wid
    y = y * img_hei  ##此处的x、y疑似错误
    h = h * img_hei
    return (x, y, w, h)


# %%
def convert_annotation(anno_dir, anno_xml, labels_dir):  ##对单个图片的anno做处理，映射为txt
    """把图像的xml文件转换为目标检测的label文件(txt)：(class,x,y,w,h)
	其中包含物体的类别，bbox的中心点坐标和宽高
	并将四个物理量归一化"""
    """
	anno_dir (str): 注释文件所在的目录路径。  
	image_id (str): 图像文件的名称（包括扩展名），例如 '000001.jpg'。  
	labels_dir (str): 转换后的标签文件保存的目录路径。  
	"""

    address = os.path.join(anno_dir, anno_xml)
    image_num=anno_xml.split(".")[0]
    in_file = open(address)
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)  ## 图片的总宽总高

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if obj.find('difficult'):
        #     difficult = int(obj.find('difficult').text)
        # else:
        #     difficult = 0
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                  float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), points)  # 返回(x,y,w,h)
        with open(os.path.join(labels_dir, '%s.txt' % (image_num)), 'a') as out_file:
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# %%
# convert_annotation(r"H:\b\CV\yolov1\VOC2007\Annotations","000005.jpg",
#                    r"H:\b\CV\yolov1\VOC2007\Label")


# %%
def make_label_txt(anno_dir, labels_dir):  ##将所有的全部转换
    """在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)
        print("make dir{}".format(labels_dir))
    filenames = os.listdir(anno_dir)
    for file in filenames:
        convert_annotation(anno_dir, file, labels_dir)


# %%
# make_label_txt(r"H:\b\CV\yolov1\VOC2007\Annotations",r"H:\b\CV\yolov1\VOC2007\labels")
#        ##               anno_dir                              labels_dir

#%%
def show_labels_img(imgname):
    # imgname是输入图像的名称，无下标
    img_path = os.path.join(r"H:/b/CV/yolov1/VOC2007/JPEGImages/",imgname)
    img=cv2.imread(img_path)
    # opencv读入的图片格式为(h,w,c)
    h, w = img.shape[:2]
    print(w, h)
    label = []
    with open(r"H:/b/CV/yolov1/VOC2007/labels/" + imgname.split(".")[0] + ".txt", 'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            print(CLASSES[int(label[0])])
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
            cv2.putText(img, CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.rectangle(img, pt1, pt2, (0, 0, 255, 2))

    cv2.imshow("img", img)
    cv2.waitKey(0)

# #%%
# show_labels_img("000005.jpg")


# %%
def img_augument(img_dir, save_img_dir, labels_dir):
    """
	:param img_dir:        原图片目录
	:param save_img_dir:   保存增强后的图片
	:param labels_dir:     存储图片信息 1.存储图片名称，用于图片的修改
									  2.augment之后的修改也会同步到其中 
	:return: 
	"""""

    imgs_list = [x.split('.')[0] + ".jpg" for x in os.listdir(labels_dir)]
    for img_name in imgs_list:
        print("process %s" % os.path.join(img_dir, img_name))
        img = cv2.imread(os.path.join(img_dir, img_name))  ##此处的img是hwc结构
        h, w = img.shape[0:2]
        input_size = 448  # 输入YOLOv1网络的图像尺寸为448x448
        # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
        # 然后再将Padding后的正方形图像缩放成448x448
        padw, padh = 0, 0  # 要记录宽高方向的padding具体数值，因为padding之后需要调整bbox的位置信息

        # 根据图像的高度和宽度，决定是否需要padding以及padding的大小
        # 目标是将图像padding成正方形
        if h > w:
            padw = (h - w) // 2
            img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=0)
        elif w > h:
            padh = (w - h) // 2
            img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)

        # 将正方形resize为 448*448大小
        img = cv2.resize(img, (input_size, input_size))
        cv2.imwrite(os.path.join(save_img_dir, img_name), img)
        # 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
        with open(os.path.join(labels_dir, img_name.split('.')[0] + ".txt"), 'r') as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if len(bbox) % 5 != 0:
            raise ValueError("File:"
                             + os.path.join(labels_dir, img_name.split('.')[0] + ".txt") + "——bbox Extraction Error!")

        # 根据padding、图像增广等操作，将原始的bbox数据转换为修改后图像的bbox数据
        if padw != 0:
            for i in range(len(bbox) // 5):
                #这步操作计算的没问题，加一个左边的，再除总的
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
                if STATIC_DEBUG:
                    cv2.rectangle(img, (int(bbox[1] * input_size - bbox[3] * input_size / 2),
                                        int(bbox[2] * input_size - bbox[4] * input_size / 2)),
                                  (int(bbox[1] * input_size + bbox[3] * input_size / 2),
                                   int(bbox[2] * input_size + bbox[4] * input_size / 2)), (0, 0, 255))
        ##  (cls, xc, yc, w, h)
        elif padh != 0:
            for i in range(len(bbox) // 5):
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w  ##补充到和w一般大，所以用w除
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w
                if STATIC_DEBUG:
                    cv2.rectangle(img, (int(bbox[1] * input_size - bbox[3] * input_size / 2),
                                        int(bbox[2] * input_size - bbox[4] * input_size / 2)),
                                  (int(bbox[1] * input_size + bbox[3] * input_size / 2),
                                   int(bbox[2] * input_size + bbox[4] * input_size / 2)), (0, 0, 255))
        # 此处可以写代码验证一下，查看padding后修改的bbox数值是否正确，在原图中画出bbox检验
        if STATIC_DEBUG:
            cv2.imshow("bbox-%d" % int(bbox[0]), img)
            cv2.waitKey(0)
        # open with ”w“ will delete all str in txt
        with open(os.path.join(labels_dir, img_name.split('.')[0] + ".txt"), 'w') as f:
            for i in range(len(bbox) // 5):
                sub_bbox = [str(x) for x in bbox[i * 5:(i * 5 + 5)]]
                str_context = " ".join(sub_bbox) + '\n'
                f.write(str_context)

# #%%
# img_augument(img_dir=r"H:\b\CV\yolov1\VOC2007\JPEGImages",save_img_dir=r"H:\b\CV\yolov1\VOC2007\Augimage",
#              labels_dir=r"H:\b\CV\yolov1\VOC2007\Labels")


#%%
def convert_bbox2labels(bbox):
    # 对单个bbox的转换
    # 转换的结果即 (7,7,5*B+cls_num)
    """将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
	注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式"""
    gridsize = 1.0 / GL_NUMGRID  # 此处为7个格。表示一个小格的长度（总长为1）  之后除girdsize就是转换为小格制
    labels = np.zeros((7, 7, 5 * GL_NUMBBOX + len(CLASSES)))
    for i in range(len(bbox) // 5):
        gridx = int(bbox[i * 5 + 1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
        gridy = int(bbox[i * 5 + 2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        gridpx = bbox[i * 5 + 1] / gridsize - gridx   #这里和下边一行，都转成了小格制，再减整数，得到在小格中的位置
        gridpy = bbox[i * 5 + 2] / gridsize - gridy
        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        # 此处的宽高使用的是对整体的比例 x,y使用的是占小格的比例
        labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10 + int(bbox[i * 5])] = 1  # 大概算独热
        # (0-4):  x y w h 1
        # (5-9):  x y w h 1
    labels = labels.reshape(1, -1)   ##此处的返回结果是(1470,)之后应该在某步转回来(此时为7,7,30)
    return labels

#%%
def create_csv_txt(img_dir, anno_dir, labels_dir, save_root_dir, save_img_dir,
                   train_val_ratio=0.9, padding=10, debug=False):
    """
    训练集与测试集的大小在此处决定，调用之前的所有函数，得到存储图片名的train/test.txt
                                                得到存储标记的 trian/test.csv
	TODO:
	将img_dir文件夹内的图片按实际需要处理后，存入save_img_dir
	最终得到图片文件夹及所有图片对应的标注(train.csv/test.csv)和图片列表文件(train.txt, test.txt)
	"""
    """
	img_dir        存储图片
	anno_dir       存储图片的描述信息
	labels_dir     存储处理过后的图片描述信息
	save_root_dir  存储train/test.txt  即训练集/测试集图片的名称
	save_img_dir   存储augment之后的图片
	"""
    # labels_dir为平级地址，其中以txt形式存储每张图片的obj的信息
    # if not os.path.exists(labels_dir):
    #     os.mkdir(labels_dir)
    #     make_label_txt(anno_dir, labels_dir)
    #     print("labels done.")
    #
    # # save_img_dir为平级地址，存储augment过后的图片
    # if not os.path.exists(save_img_dir):
    #     os.mkdir(save_img_dir)
    #     img_augument(img_dir, save_img_dir, labels_dir)
    #     print("augment done")

    # save_root_dir 在train/test.txt中存储训练集/测试集所有图片的名称
    if not os.path.exists(save_root_dir):
        os.mkdir(save_root_dir)

    imgs_list = os.listdir(save_img_dir)  # 列出所有图片的名字
    n_trainval = len(imgs_list)
    shuffle_id = list(range(n_trainval))  # 0-len，打乱后用于抽取
    random.shuffle(shuffle_id)
    n_train = int(n_trainval * train_val_ratio)  # 用于截断 shuffle_id

    train_id = shuffle_id[:n_train]  # 抽取训练集
    test_id = shuffle_id[n_train:]  # 抽取测试集

#无论是train还是test，返回的都是 n_train/test，-1,之后都要重新恢复维度
#处理训练集
    # trian.txt中的图片名和train.csv中的图片标记 是一一对应的
    traintxt = open(os.path.join(save_root_dir, "train.txt"), 'w')  # 存储训练集中所有图片的名字
    # 写入训练集所有项的名字
    traincsv = np.zeros((n_train, GL_NUMGRID * GL_NUMGRID * (5 * GL_NUMBBOX + len(CLASSES))), dtype=np.float32)
    # 训练集中的所有信息都会存入traincsv中
    for i, id in enumerate(train_id):  # i标识是第几个，id为index
        img_name = imgs_list[id]       # id对应的图片的名字
        img_name =img_name+"\n"
        # img_path = os.path.join(save_img_dir, img_name) + '\n'  # save_img_dir建议给出绝对地址
        traintxt.write(img_name)
        with open(os.path.join(labels_dir, "%s.txt" % img_name.split('.')[0]), 'r') as f:
            bbox = [float(x) for x in f.read().split()]
            traincsv[i, :] = convert_bbox2labels(bbox)  # traincsv.shape(batchsize,1470) (7,7,30)→1470
                                                        # 读出之前存储在txt中的bbox信息，转化为Yolo
                                                        # 的输出格式，存入csv中,每次存储一张图片的信息

    np.savetxt(os.path.join(save_root_dir, "train.csv"), traincsv)
    print("Create %d train data." % (n_train))

# 处理测试集
    testtxt = open(os.path.join(save_root_dir, "test.txt"), 'w')
    testcsv = np.zeros((n_trainval - n_train, GL_NUMGRID * GL_NUMGRID * (5 * GL_NUMBBOX + len(CLASSES))),
                       dtype=np.float32)
    for i, id in enumerate(test_id):
        img_name = imgs_list[id]
        img_name =img_name+"\n"
        testtxt.write(img_name)
        with open(os.path.join(labels_dir, "%s.txt" % img_name.split('.')[0]), 'r') as f:
            bbox = [float(x) for x in f.read().split()]
            testcsv[i, :] = convert_bbox2labels(bbox)        # 1470==7*7*30
    np.savetxt(os.path.join(save_root_dir, "test.csv"), testcsv)
    print("Create %d test data." % (n_trainval - n_train))


# #%%
# create_csv_txt(img_dir="./VOC2007/JPEGImages",anno_dir="./VOC2007/Annotations",labels_dir="./VOC2007/labels",
#                save_root_dir="./VOC2007/root",save_img_dir="./VOC2007/Augimage")

