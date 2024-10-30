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




# #%%
# net=MyNet()
# X = torch.randn(size=(64, 3, 448, 448))
# net(X)


# #%%
# print(torch.cuda.is_available())
import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())



# #%%
# from net import MyNet
# net=MyNet()
# pred=torch.rand((64,1470))
# label=torch.zeros((64,1470))
# metric=net.calculate_metric(preds=pred,labels=label)
# print(metric)


# #%%
# model=torch.load("./check_point/epoch100.pkl")
# #%%
# print(next(model.parameters()).device)
