# Laboro-Tomato-Mask-RCNN

本项目基于 Mask R-CNN 模型，对番茄数据集进行目标检测与熟度分类。项目使用 Python、TensorFlow 和 Keras 实现。

## 项目功能
- **目标检测**：识别图像中的番茄。
- **熟度分类**：对检测到的番茄进行熟度等级分类。

## 环境要求
- **Python 版本**：Python 3.7
- **框架**：
  - TensorFlow 2.0.0
  - Keras 2.3.1
- 所需依赖见 `requirements.txt` 文件，可通过以下命令安装：
  ```bash
  pip install -r requirements.txt
  ```

## 数据集
- 数据集来源：[Laboro Tomato Dataset](https://datasetninja.com/laboro-tomato)。
- 本项目中包含了一个样本数据集用于测试，如需完整数据集，请从原数据集网站下载。

## 预训练权重
- 训练模型时需要使用预训练权重。请从以下链接下载：
  [Mask R-CNN TF2 预训练权重](https://github.com/ahmedfgad/Mask-RCNN-TF2/releases/download/v3.0/mask_rcnn_coco.h5)。

## 使用方法

### 模型训练
1. 将下载的预训练权重放置在项目目录中。
2. 运行训练脚本：
   ```bash
   python tomato_training.py
   ```
3. 训练完成后，权重文件将保存为 `LaboroTomato_mask_rcnn_trained.h5`。

### 模型预测
1. 使用训练好的权重进行检测和分类：
   ```bash
   python tomato_prediction.py
   ```

## 项目结构
- **`mrcnn/`**：包含 Mask R-CNN 模型代码。
  - 支持 TensorFlow 2.0.0 和 Keras 2.2.4 或 2.3.1。
  - 支持 Python 3.7.3（兼容 Python 3.6.9 和 3.6.13）。
  - **注意**：不支持 TensorFlow 1.x。
  - **`tomato_training.py`**：用于训练模型的脚本，包含数据加载、模型训练和回调设置。
  - **`tomato_prediction.py`**：用于对测试数据进行预测的脚本。
  - **`requirements.txt`**：项目的依赖包列表，确保你安装了正确的环境。
  - **`tomato_training_wandb.py`**：在原`tomato_training.py`基础上集成了 Weights & Biases (Wandb) 相关代码。

## 参考资料
- Mask R-CNN TensorFlow 2 版本实现：[Ahmed Gad's Mask R-CNN](https://github.com/ahmedfgad/Mask-RCNN-TF2)
- 数据集：[Laboro Tomato Dataset](https://datasetninja.com/laboro-tomato)

## 其他信息
1. **Wandb 集成**：项目集成了 [Weights & Biases](https://wandb.ai/)，用于实时记录和可视化训练过程。你可以通过 Wandb 仪表盘实时查看每次运行的终端输出、训练中的图像、损失值、准确率等指标，最终权重会保存在平台中以便下载。
2. **模型性能**：该项目适合初学者用于学习 Mask R-CNN 和目标检测任务。由于数据集较为简单，它可以帮助你快速了解如何使用 Mask R-CNN 进行目标检测与分类任务。

## 联系方式
如果遇到任何问题或有建议，请在 [GitHub](https://github.com/MilleXi/Laboro-Tomato-Mask-RCNN) 提交 Issue 或 PR，感谢你的参与！
