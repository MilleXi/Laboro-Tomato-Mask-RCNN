import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import json

# 定义类名列表，包含LaboroTomato数据集中的类别
CLASS_NAMES = ['BG', 'b_green', 'l_green', 'l_fully_ripened', 'b_half_ripened', 'l_half_ripened', 'b_fully_ripened']

# 定义Mask R-CNN的配置类
class SimpleConfig(mrcnn.config.Config):
    # 配置名称，用于标识配置
    NAME = "laboro_tomato_inference"
    
    # 设置GPU数量和每个GPU处理的图片数量
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 类别数量（包含背景类）
    NUM_CLASSES = len(CLASS_NAMES)

# 初始化Mask R-CNN模型并加载预训练权重
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# 加载模型权重
model.load_weights(filepath="mask_rcnn_final.h5", 
                   by_name=True)

# 定义输入图片路径和标注文件路径
image_folder = "laboro-tomato-DatasetNinja/Test/img"
annotation_folder = "laboro-tomato-DatasetNinja/Test/ann"
image_name = "IMG_0991.jpg"
annotation_name = image_name + ".json"

# 读取图片并进行预处理（从BGR转换为RGB）
image_path = os.path.join(image_folder, image_name)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 检查对应的标注文件是否存在
annotation_path = os.path.join(annotation_folder, annotation_name)
if not os.path.exists(annotation_path):
    print(f"标注文件 {annotation_path} 不存在！")
else:
    # 加载标注文件（JSON格式）并输出内容
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)
        print("标注文件内容：", annotation_data)

# 使用模型检测图片中的物体
results = model.detect([image], verbose=0)
r = results[0]
os.makedirs('pic',exist_ok=True)
# 可视化检测结果
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'],
                                  save_fig_path='pic')

# 打印检测结果
print("检测结果：")
for i in range(len(r['class_ids'])):
    class_id = r['class_ids'][i]
    class_name = CLASS_NAMES[class_id]
    box = r['rois'][i]
    score = r['scores'][i]
    print(f"类别: {class_name}, 置信度: {score:.2f}, 边界框: {box}")
