import os
import json
from numpy import zeros, asarray
import mrcnn.utils
import mrcnn.config
import mrcnn.model

class LaboroTomatoDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # 添加类别信息
        self.add_class("dataset", 1, "b_green")
        self.add_class("dataset", 2, "l_green")
        self.add_class("dataset", 3, "l_fully_ripened")
        self.add_class("dataset", 4, "b_half_ripened")
        self.add_class("dataset", 5, "l_half_ripened")
        self.add_class("dataset", 6, "b_fully_ripened")

        images_dir = os.path.join(dataset_dir, 'img')
        annotations_dir = os.path.join(dataset_dir, 'ann')

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]  # 去掉文件扩展名

            # 根据图片ID分割训练和验证集
            if is_train and int(image_id.split('_')[-1]) >= 150:
                continue
            if not is_train and int(image_id.split('_')[-1]) < 150:
                continue

            img_path = os.path.join(images_dir, filename)
            ann_path = os.path.join(annotations_dir, filename + '.json')

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotation_path = info['annotation']
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        h, w = data['imageHeight'], data['imageWidth']
        masks = zeros([h, w, len(data['shapes'])], dtype='uint8')

        class_ids = []
        for i, shape in enumerate(data['shapes']):
            points = shape['points']
            class_name = shape['label']
            if class_name not in self.class_names:
                continue
            class_id = self.class_names.index(class_name)

            # 创建掩码
            rr, cc = self.polygon_to_mask(points, h, w)
            masks[rr, cc, i] = 1
            class_ids.append(class_id)

        return masks, asarray(class_ids, dtype='int32')

    def polygon_to_mask(self, points, height, width):
        from skimage.draw import polygon
        y_points, x_points = zip(*points)
        rr, cc = polygon(y_points, x_points, (height, width))
        return rr, cc

class LaboroTomatoConfig(mrcnn.config.Config):
    NAME = "laboro_tomato_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 7  # 6 类别 + 背景

    STEPS_PER_EPOCH = 500

# 加载训练数据集
train_dataset = LaboroTomatoDataset()
train_dataset.load_dataset(dataset_dir='laboro-tomato-DatasetNinja/Train', is_train=True)
train_dataset.prepare()

# 加载验证数据集
validation_dataset = LaboroTomatoDataset()
validation_dataset.load_dataset(dataset_dir='laboro-tomato-DatasetNinja/Test', is_train=False)
validation_dataset.prepare()

# 模型配置
laboro_tomato_config = LaboroTomatoConfig()

# 构建 Mask R-CNN 模型
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=laboro_tomato_config)

# 加载 COCO 预训练权重并排除不需要的头部
model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# 训练模型
model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=laboro_tomato_config.LEARNING_RATE, 
            epochs=10, 
            layers='heads')

# 保存训练好的权重
model_path = 'LaboroTomato_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
