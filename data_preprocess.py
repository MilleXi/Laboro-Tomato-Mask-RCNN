import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import json
from PIL import Image
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TomatoDataset(Dataset):
    def __init__(self, data, transform=None, train=True):
        self.data = data
        self.train = train

        self.class_to_idx = {
            'b_green': 0, 
            'l_green': 1, 
            'l_fully_ripened': 2,
            'b_half_ripened': 3, 
            'l_half_ripened': 4,
            'b_fully_ripened': 5,
        }

        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((800, 800), antialias=True), 
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            if train:
                self.aug_transform = T.Compose([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.3),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    T.RandomRotation(30)
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def calculate_bbox(self, polygon_points):
        """计算多边形的边界框"""
        x = polygon_points[:, 0]
        y = polygon_points[:, 1]
        return [float(x.min()), float(y.min()), float(x.max()), float(y.max())]

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image_path']
        ann_path = item['ann_path']

        logger.info(f"Processing image {idx + 1}/{len(self.data)}: {img_path}")

        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"Unable to read image: {img_path}")
            raise ValueError(f"Unable to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # 初始化掩码列表和其他目标信息
        masks = []
        boxes = []
        labels = []

        with open(ann_path, 'r') as f:
            ann_data = json.load(f)

        for obj in ann_data['objects']:
            if obj['geometryType'] == 'polygon':
                # 获取多边形点
                polygon_points = np.array(obj['points']['exterior'], dtype=np.int32)
                
                # 创建单个对象的掩码
                obj_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(obj_mask, [polygon_points], 1)
                masks.append(obj_mask)

                # 计算边界框
                bbox = self.calculate_bbox(polygon_points)
                boxes.append(bbox)

                # 获取类别标签
                class_title = obj['classTitle']
                class_idx = self.class_to_idx.get(class_title)
                if class_idx is not None:
                    labels.append(class_idx)

        # 确保至少有一个目标
        if not masks:
            # 如果没有目标，创建虚拟数据
            masks = [np.zeros((height, width), dtype=np.uint8)]
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]

        # 转换为张量
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 计算区域
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        # 构建目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return image, target
    
def visualize(image, target):
    """可视化函数用于调试"""
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    
    # 转换图像格式从 Tensor [C, H, W] 到 [H, W, C]
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)  # 确保值在有效范围内

    # 获取目标信息
    boxes = target['boxes'].cpu().numpy()
    labels = target['labels'].cpu().numpy()
    masks = target['masks'].cpu().numpy()

    # 创建图像
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # 绘制每个实例
    for box, label, mask in zip(boxes, labels, masks):
        # 绘制掩码
        mask_image = np.ma.masked_where(mask == 0, mask)
        plt.imshow(mask_image, alpha=0.5, cmap='jet')

        # 绘制边界框
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        plt.gca().add_patch(rect)

        # 添加类别标签
        plt.text(x1, y1, f'Class {label}', 
                bbox=dict(facecolor='white', alpha=0.7))

    plt.axis('off')
    plt.show()

# 使用示例
# dataset = TomatoDataset(train_data, train=True)
# image, target = dataset[0]
# visualize(image, target)