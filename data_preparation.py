import os
import json
import numpy as np
from PIL import Image
import torch
import cv2

def process_dataset(base_path):
    train_path = os.path.join(base_path, "Train")
    test_path = os.path.join(base_path, "Test")
    
    def process_split(split_path):
        ann_path = os.path.join(split_path, "ann")
        img_path = os.path.join(split_path, "img")
        data = []
        
        for ann_file in os.listdir(ann_path):
            if not ann_file.endswith('.json'):
                continue
                
            img_name = ann_file.replace('.json', '')
            img_file = os.path.join(img_path, img_name)
            
            if not os.path.exists(img_file):
                print(f"Warning: Missing image for {ann_file}")
                continue
            
            # 仅读取图像尺寸而不加载整个图像
            with Image.open(img_file) as img:
                width, height = img.size
            
            # 延迟加载标注和掩码数据
            item_data = {
                'image_path': img_file,
                'ann_path': os.path.join(ann_path, ann_file),
                'height': height,
                'width': width
            }
            data.append(item_data)
        return data

    return process_split(train_path), process_split(test_path)

# 使用示例
# train_data, test_data = process_dataset("laboro-tomato-DatasetNinja")