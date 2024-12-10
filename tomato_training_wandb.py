import os
import json
from numpy import zeros, asarray
import mrcnn.utils
import mrcnn.config
import mrcnn.model
import wandb
import tensorflow as tf
import keras
from keras import backend as K
from datetime import datetime
from skimage.draw import polygon

# 初始化 wandb，使用固定的 run 名称
run_name = f"tomato-detection-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
wandb.init(
    project="laboro-tomato-detection",
    name=run_name,  # 使用固定的运行名称
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 1,
        "backbone": "resnet50",
        "rpn_nms_threshold": 0.7,
        "rpn_train_anchors": 256,
        "post_nms_rois_training": 2000,
    }
)

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
            image_id = filename[:-4]

            if is_train and int(image_id.split('_')[-1]) >= 150:
                continue
            if not is_train and int(image_id.split('_')[-1]) < 150:
                continue

            img_path = os.path.join(images_dir, filename)
            ann_path = os.path.join(annotations_dir, filename + '.json')

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        # 获取图片的信息
        info = self.image_info[image_id]
        annotation_path = info['annotation']
        
        # 读取标注文件
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        # 获取图片的高度和宽度
        h, w = data['size']['height'], data['size']['width']
        
        # 初始化掩码
        masks = zeros([h, w, len(data['objects'])], dtype='uint8')
        
        class_ids = []
        
        # 遍历所有标注对象
        for i, obj in enumerate(data['objects']):
            # 获取类别名称
            class_name = obj['classTitle']
            if class_name not in self.class_names:
                continue
            class_id = self.class_names.index(class_name)
            
            # 提取多边形点
            points = obj['points']['exterior']
            rr, cc = self.polygon_to_mask(points, h, w)
            
            # 将掩码更新为1
            masks[rr, cc, i] = 1
            class_ids.append(class_id)
        
        # 返回掩码和类别ID数组
        return masks, asarray(class_ids, dtype='int32')

    def polygon_to_mask(self, points, height, width):
        """
        将多边形点转换为掩码
        :param points: 多边形点列表 [(x1, y1), (x2, y2), ...]
        :param height: 图片高度
        :param width: 图片宽度
        :return: 掩码的行列坐标
        """
        y_points, x_points = zip(*points)
        rr, cc = polygon(y_points, x_points, (height, width))
        return rr, cc

class LaboroTomatoConfig(mrcnn.config.Config):
    NAME = "laboro_tomato_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 7
    STEPS_PER_EPOCH = 500
    BACKBONE = "resnet50"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    LEARNING_RATE = 0.001
    EPOCHS = 10

# 自定义 TensorBoard 回调，禁用 trace 功能
class CustomTensorBoard(keras.callbacks.TensorBoard):
    def _init_writer(self, *args, **kwargs):
        super()._init_writer(*args, **kwargs)
        # 禁用 trace，避免相关错误
        self._trace_on = False
        
    def on_train_batch_end(self, batch, logs=None):
        # 覆盖原方法，避免调用 trace 相关功能
        if self.update_freq == 'batch':
            self._write_logs(logs, self._train_step)

# 自定义 WandB 回调
class WandBCallback(keras.callbacks.Callback):
    def __init__(self, log_dir=None):
        super().__init__()
        self.log_dir = log_dir
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = tf.timestamp()
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # 记录所有损失
        wandb.log({
            'epoch': epoch,
            'learning_rate': float(K.get_value(self.model.optimizer.learning_rate)) if hasattr(self.model.optimizer, 'learning_rate') else float(K.get_value(self.model.optimizer.lr)),
            'total_loss': float(logs.get('loss', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'epoch_time': float(tf.timestamp() - self.epoch_start_time)
        })
        
        if self.log_dir:
            model_path = os.path.join(self.log_dir, f'mask_rcnn_epoch_{epoch:04d}.h5')
            if os.path.exists(model_path):
                wandb.save(model_path, base_path=self.log_dir)
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if batch % 100 == 0:  # 每100个batch记录一次
            wandb.log({
                'batch_total_loss': float(logs.get('loss', 0)),
            })

def train_model():
    try:
        # 加载数据集
        train_dataset = LaboroTomatoDataset()
        train_dataset.load_dataset(dataset_dir='laboro-tomato-DatasetNinja/Train', is_train=True)
        train_dataset.prepare()

        validation_dataset = LaboroTomatoDataset()
        validation_dataset.load_dataset(dataset_dir='laboro-tomato-DatasetNinja/Test', is_train=False)
        validation_dataset.prepare()

        # 配置和初始化模型
        config = LaboroTomatoConfig()
        model = mrcnn.model.MaskRCNN(mode='training', 
                                    model_dir='./', 
                                    config=config)

        # 加载预训练权重
        model.load_weights(filepath='mask_rcnn_coco.h5', 
                          by_name=True, 
                          exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        # 设置回调
        callbacks = [
            WandBCallback(log_dir=model.log_dir),
            CustomTensorBoard(log_dir=model.log_dir,
                            write_graph=True,
                            update_freq='epoch'),
            keras.callbacks.ModelCheckpoint(
                os.path.join(model.log_dir, "mask_rcnn_{epoch:04d}.h5"),
                save_weights_only=True
            ),
        ]

        # 训练模型
        model.train(train_dataset=train_dataset, 
                    val_dataset=validation_dataset, 
                    learning_rate=config.LEARNING_RATE, 
                    epochs=config.EPOCHS, 
                    layers='heads',
                    custom_callbacks=callbacks)
        
        # 训练结束后保存最终模型
        final_model_path = os.path.join(model.log_dir, 'mask_rcnn_final.h5')
        model.keras_model.save_weights(final_model_path)
        wandb.save(final_model_path, base_path=model.log_dir)
        
        # 记录最终的模型配置
        wandb.config.update({
            "final_training_steps": config.STEPS_PER_EPOCH * config.EPOCHS,
            "backbone": config.BACKBONE,
            "num_classes": config.NUM_CLASSES,
            "image_min_dim": config.IMAGE_MIN_DIM,
            "image_max_dim": config.IMAGE_MAX_DIM,
            "rpn_anchor_scales": config.RPN_ANCHOR_SCALES,
        })
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    train_model()