import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import matplotlib.patches as patches
from tqdm import tqdm

class TomatoDataset(Dataset):
    def __init__(self, data, transform=None, train=True):
        self.data = data
        self.train = train
        
        # 确保类别映射没有重复
        self.class_to_idx = {
            'b_green': 0, 
            'l_green': 1, 
            'l_fully_ripened': 2,
            'b_half_ripened': 3, 
            'l_half_ripened': 4,
            'b_fully_ripened': 5,
        }
        
        # 如果没有指定变换，则使用默认的变换
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((800, 800), antialias=True),  # 保证图像尺寸
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            
            # 训练集时使用增强变换
            if train:
                self.aug_transform = T.Compose([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.3),
                    T.ColorJitter(brightness=0.2,
                                  contrast=0.2,
                                  saturation=0.2),
                    T.RandomRotation(30)
                ])
        else:
            self.transform = transform

        # 如果数据集很大，加载时显示进度条
        print(f"Dataset loaded with {len(data)} items.")

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 加载图像并显示进度
        print(f"Processing image {idx + 1}/{len(self.data)}: {item['image_path']}")
        
        # 读取图像并进行基本的处理
        image = cv2.imread(item['image_path'])
        if image is None:
            raise ValueError(f"无法加载图像: {item['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取标注数据
        with open(item['ann_path'], 'r') as f:
            ann_data = json.load(f)
        
        # 处理每个对象（例如，分割掩码）
        masks = []
        for obj in ann_data['objects']:
            # 计算每个分割掩码
            points = obj['points']['exterior']
            polygon = Polygon(points)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if polygon.contains(Point(x, y)):
                        mask[y, x] = 1
            masks.append(mask)
        
        # 转换掩码为 tensor
        masks = np.array(masks)  # 使用 np.array 合并掩码列表
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # 如果是训练集，则应用增强变换
        if self.train:
            for i in range(len(masks)):
                masks[i] = self.aug_transform(masks[i])

        # 应用图像的转换
        if self.transform:
            image = self.transform(image)

        # 返回图像和目标
        target = {
            "boxes": torch.tensor([obj['points']['exterior'] for obj in ann_data['objects']], dtype=torch.float32),
            "labels": torch.tensor([self.class_to_idx[obj['classTitle']] for obj in ann_data['objects']], dtype=torch.int64),
            "masks": masks
        }

        return image, target
    

def visualize(image, target):
    # 转换图像格式从 Tensor [C, H, W] 到 [H, W, C]
    image = image.permute(1, 2, 0).cpu().numpy()

    # 获取目标框、标签和掩码信息
    boxes = target['boxes'].cpu().numpy()
    labels = target['labels'].cpu().numpy()

    # 确保掩码是一个 Tensor
    masks = target['masks']  # 如果 masks 已经是 Tensor，无需转换
    if isinstance(masks, list):
        masks = torch.as_tensor(masks, dtype=torch.uint8)
    
    # 使用 matplotlib 绘制图像
    plt.figure(figsize=(10, 10))
    plt.imshow(image)  # 直接显示图像

    # 绘制掩码
    for i in range(len(masks)):
        mask = masks[i].cpu().numpy()  # 获取掩码
        mask = np.ma.masked_where(mask == 0, mask)  # 只显示掩码部分
        plt.imshow(mask, cmap='jet', alpha=0.5)  # alpha 控制透明度

    # 绘制边界框
    for i in range(len(boxes)):
        box = boxes[i]  # 获取边界框
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='r', facecolor='none')  # 红色边框
        plt.gca().add_patch(rect)

    plt.axis('off')  # 不显示坐标轴
    plt.show()


# 使用示例
# dataset = TomatoDataset(train_data, train=True)
# image, target = dataset[0]
# visualize(image, target)