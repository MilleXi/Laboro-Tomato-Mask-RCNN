import os
import json
import numpy as np
from PIL import Image

def process_dataset(base_path):
    """
    解析数据集，包括图像路径、标注路径、多边形信息、类别等。
    """
    train_path = os.path.join(base_path, "Train")
    test_path = os.path.join(base_path, "Test")
    
    def process_split(split_path):
        ann_path = os.path.join(split_path, "ann")
        img_path = os.path.join(split_path, "img")
        data = []
        
        for ann_file in os.listdir(ann_path):
            if not ann_file.endswith('.json'):
                continue
            
            img_name = ann_file.replace('.json', '')  # 假设图像格式为 .jpg
            img_file = os.path.join(img_path, img_name)
            
            if not os.path.exists(img_file):
                print(f"Warning: Missing image for {ann_file}")
                continue
            
            # 读取图像尺寸
            with Image.open(img_file) as img:
                width, height = img.size
            
            # 解析标注文件
            with open(os.path.join(ann_path, ann_file), 'r') as f:
                ann_data = json.load(f)
            
            objects = ann_data.get("objects", [])
            annotations = []
            
            for obj in objects:
                # 检查标注是否为多边形
                if obj.get("geometryType") == "polygon":
                    points = obj["points"]["exterior"]
                    class_title = obj["classTitle"]
                    
                    # 转换多边形为 numpy 数组
                    polygon = np.array(points, dtype=np.int32)
                    
                    # 保存标注信息
                    annotations.append({
                        "class_title": class_title,
                        "polygon": polygon
                    })
            
            # 构造数据项
            item_data = {
                "image_path": img_file,
                "ann_path": os.path.join(ann_path, ann_file),
                "height": height,
                "width": width,
                "annotations": annotations  # 包含多边形和类别信息
            }
            data.append(item_data)
        return data

    return process_split(train_path), process_split(test_path)

# 使用示例
# train_data, test_data = process_dataset("laboro-tomato-DatasetNinja")

