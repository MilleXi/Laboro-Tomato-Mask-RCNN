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
import logging

# 配置日志，输出详细信息
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 输出当前处理的图像路径
        logger.info(f"Processing image {idx + 1}/{len(self.data)}: {item['image_path']}")
        
        # 延迟加载图像
        image = cv2.imread(item['image_path'])
        if image is None:
            logger.error(f"Failed to load image: {item['image_path']}")
            raise ValueError(f"Unable to load image: {item['image_path']}")
        
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取图像大小
        height, width, _ = image.shape
        
        # 创建空白掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 读取标注文件
        with open(item['ann_path'], 'r') as f:
            ann_data = json.load(f)
        
        # 解析标注
        masks = []
        areas = []
        labels = []
        
        for obj in ann_data['objects']:
            if obj['geometryType'] != 'polygon':
                continue
            
            # 提取每个物体的外轮廓
            polygon_points = obj['points']['exterior']
            polygon = Polygon(polygon_points)
            
            # 创建掩码
            mask_obj = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    if polygon.contains(Point(x, y)):
                        mask_obj[y, x] = 1
            
            masks.append(mask_obj)
            areas.append(mask_obj.sum())  # 计算每个掩码的面积
            labels.append(self.class_to_idx[obj['classTitle']])
        
        # 将掩码转为Tensor格式
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        areas = torch.tensor(areas, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'masks': masks,
            'labels': labels,
            'area': areas,
            'image_id': torch.tensor([idx]),
            'boxes': torch.tensor([[0, 0, width, height]], dtype=torch.float32)  # 用于兼容其他任务
        }
        
        if self.transform:
            image = self.transform(image)
        
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