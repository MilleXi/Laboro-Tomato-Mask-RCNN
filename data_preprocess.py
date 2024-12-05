import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import json
from PIL import Image

class TomatoDataset(Dataset):
    def __init__(self, data, transform=None, train=True):
        self.data = data
        self.train = train
        self.class_to_idx = self.class_to_idx = {
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
                T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ])
            
            if train:
                self.aug_transform = T.Compose([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.3),
                    T.ColorJitter(brightness=0.2,
                                contrast=0.2,
                                saturation=0.2),
                    T.RandomRotation(30)
                ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 延迟加载图像
        image = cv2.imread(item['image_path'])
        if image is None:
            raise ValueError(f"无法加载图像: {item['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 延迟加载和处理标注
        with open(item['ann_path']) as f:
            ann_data = json.load(f)
            
        masks = []
        labels = []
        boxes = []
        
        for obj in ann_data.get('objects', []):
            points = np.array(obj['points']['exterior'], dtype=np.int32)
            class_name = obj['classTitle']
            
            # 生成掩码
            mask = np.zeros((item['height'], item['width']), dtype=np.uint8)
            cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 1)
            masks.append(mask)
            
            # 获取边界框
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            boxes.append([x_min, y_min, x_max, y_max])
            
            labels.append(self.class_to_idx[class_name])
        
        # 转换为tensor
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        
        # 数据增强
        image = Image.fromarray(image)
        if self.train and np.random.rand() > 0.5:
            image = self.aug_transform(image)
        image = np.array(image)
        image = self.transform(image)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }
        
        return image, target

# 创建数据集
# train_dataset = TomatoDataset(train_data, train=True)
# test_dataset = TomatoDataset(test_data, train=False)