import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import wandb
import cv2
import logging
import time
import torch.nn.functional as F
from data_preparation import process_dataset
from data_preprocess import TomatoDataset
from maskrcnn_model import EnhancedMaskRCNN

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    def __init__(self, model, train_dataset, val_dataset, batch_size=2, num_workers=4, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        self.model = model
        self.model.to(self.device)
        logger.info("模型已加载到设备")
        
        logger.info("初始化数据加载器...")

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        
        # 优化器设置
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.0001,
            weight_decay=0.0001
        )
        logger.info("优化器已初始化")
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.1
        )
        logger.info("学习率调度器已初始化")
        
        # 初始化wandb
        wandb.init(project="tomato-detection")
        logger.info("Wandb已初始化")
        
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        epoch_start_time = time.time()
        batch_times = []
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                batch_start_time = time.time()
                
                # 转换为tensor
                images = torch.stack(images).to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                
                total_loss += losses.item()
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                pbar.set_postfix({'loss': f'{losses.item():.4f}', 'avg_batch_time': f'{np.mean(batch_times):.2f}s'})
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {losses.item():.4f}")
            
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        metrics = {'precision': [], 'recall': [], 'f1': [], 'iou': []}
        val_start_time = time.time()
        
        logger.info("开始验证...")
        for batch_idx, (images, targets) in enumerate(tqdm(self.val_loader)):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} 
                      for t in targets]
            
            outputs = self.model(images)
            
            # 计算验证指标
            for pred, target in zip(outputs, targets):
                batch_metrics = self.calculate_metrics(pred, target)
                for k, v in batch_metrics.items():
                    metrics[k].append(v)
            
            if batch_idx % 10 == 0:
                logger.info(f"已处理 {batch_idx}/{len(self.val_loader)} 批次")
        
        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        val_time = time.time() - val_start_time
        
        logger.info(
            f"验证完成 - 用时: {val_time:.2f}s\n"
            f"平均IoU: {avg_metrics['iou']:.4f}\n"
            f"平均精确率: {avg_metrics['precision']:.4f}\n"
            f"平均召回率: {avg_metrics['recall']:.4f}\n"
            f"平均F1分数: {avg_metrics['f1']:.4f}"
        )
        
        wandb.log(avg_metrics)
        return avg_metrics
    
    def visualize_features(self, image):
        """可视化特征图"""
        # 获取backbone的特征图
        features = self.model.model.backbone(image.unsqueeze(0))
        
        for level, feature_map in features.items():
            # 转换为热力图
            feature_map = feature_map.mean(dim=1)
            heatmap = feature_map.squeeze().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # 记录到wandb
            wandb.log({f"feature_map_level_{level}": 
                      wandb.Image(heatmap)})
    
    def visualize_predictions(self, images, predictions, targets):
        """可视化预测结果"""
        fig, axes = plt.subplots(len(images), 3, figsize=(15, 5*len(images)))
        
        for i, (img, pred, target) in enumerate(zip(images, predictions, targets)):
            # 原图
            axes[i,0].imshow(img.cpu().permute(1,2,0))
            axes[i,0].set_title('Original Image')
            
            # 预测掩码
            pred_mask = pred['masks'][0].cpu().squeeze()
            axes[i,1].imshow(pred_mask)
            axes[i,1].set_title('Predicted Mask')
            
            # 真实掩码
            target_mask = target['masks'][0].cpu()
            axes[i,2].imshow(target_mask)
            axes[i,2].set_title('Ground Truth')
            
        plt.tight_layout()
        wandb.log({"predictions": wandb.Image(plt)})
        plt.close()
    
    def box_iou(box1, box2):
        # 计算边界框IoU
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        return inter / union

    def recall_by_area(pred_masks, target_masks, area_mask):
        """计算特定尺度目标的召回率"""
        filtered_target = target_masks[area_mask]
        filtered_pred = pred_masks[area_mask]
        intersection = (filtered_pred & filtered_target).sum().float()
        return intersection / (filtered_target.sum() + 1e-6)
    
    def calculate_metrics(self, pred, target):
        if len(pred['masks']) == 0:  # 处理无预测的情况
            return {
                'iou': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'ap50': 0.0,
                'ap75': 0.0,
                'ap85': 0.0,
                'map': 0.0
            }

        pred_masks = pred['masks'].squeeze(1) > 0.5
        target_masks = target['masks']
        if len(target_masks.shape) == 3:
            target_masks = target_masks.unsqueeze(1)
        
        # 确保batch维度匹配
        min_size = min(len(pred_masks), len(target_masks))
        pred_masks = pred_masks[:min_size]
        target_masks = target_masks[:min_size]
        
        target_masks = F.interpolate(target_masks.float(), 
                                size=pred_masks.shape[-2:],
                                mode='nearest').squeeze(1).bool()
        
        pred_boxes = pred['boxes']
        target_boxes = target['boxes']
        pred_scores = pred['scores']

        # 基础指标
        intersection = (pred_masks & target_masks).sum().float()
        union = (pred_masks | target_masks).sum().float()
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        precision = intersection / (pred_masks.sum() + 1e-6)
        recall = intersection / (target_masks.sum() + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        # 计算AP@不同IOU阈值
        iou_thresholds = [0.5, 0.75, 0.85]
        ap_scores = []
        for iou_threshold in iou_thresholds:
            tp = 0
            fp = 0
            for i, pred_box in enumerate(pred_boxes):
                ious = self.box_iou(pred_box.unsqueeze(0), target_boxes)
                max_iou = ious.max()
                if max_iou >= iou_threshold and pred_scores[i] > 0.5:
                    tp += 1
                else:
                    fp += 1
            ap = tp / (tp + fp + 1e-6)
            ap_scores.append(ap)
        
        # 计算目标尺度敏感度
        area_ranges = {
            'small': (0, 32*32),
            'medium': (32*32, 96*96),
            'large': (96*96, float('inf'))
        }
        scale_metrics = {}
        for name, (min_area, max_area) in area_ranges.items():
            mask_areas = (target_masks.sum(dim=(1,2)) > min_area) & (target_masks.sum(dim=(1,2)) <= max_area)
            if mask_areas.any():
                scale_metrics[f'{name}_recall'] = self.recall_by_area(pred_masks, target_masks, mask_areas)
        
        return {
            'iou': iou.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'ap50': ap_scores[0],
            'ap75': ap_scores[1],
            'ap85': ap_scores[2],
            'map': sum(ap_scores) / len(ap_scores),
            **scale_metrics
        }

    def train(self, num_epochs):
        logger.info(f"开始训练 - 总轮数: {num_epochs}")
        best_f1 = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            logger.info(f"\n开始 Epoch {epoch}/{num_epochs-1}")
            
            train_loss = self.train_one_epoch(epoch)
            metrics = self.validate()
            
            # 更新学习率
            self.scheduler.step(train_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 保存最佳模型
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                save_path = f'best_model_epoch_{epoch}_f1_{best_f1:.4f}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': best_f1,
                }, save_path)
                logger.info(f"保存最佳模型到: {save_path}")
            
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch} 总结:\n"
                f"训练损失: {train_loss:.4f}\n"
                f"验证F1分数: {metrics['f1']:.4f}\n"
                f"当前学习率: {current_lr:.6f}\n"
                f"用时: {epoch_time:.2f}s"
            )
        
        total_time = time.time() - start_time
        logger.info(
            f"\n训练完成!\n"
            f"总用时: {total_time/3600:.2f}小时\n"
            f"最佳F1分数: {best_f1:.4f}"
        )

def main():
    logger.info("初始化训练...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 6
    model = EnhancedMaskRCNN(num_classes=num_classes)

    train_data, test_data = process_dataset("laboro-tomato-DatasetNinja")
    train_dataset = TomatoDataset(train_data, train=True)
    test_dataset = TomatoDataset(test_data, train=False)
    trainer = ModelTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        batch_size=2,
        device=device
    )
    trainer.train(num_epochs=50)

if __name__ == '__main__':
    main()