import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

class EnhancedMaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 基础模型使用ResNet50-FPN作为backbone
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        
        # 改进1: 多尺度RoI对齐
        self.model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        
        # 改进2: 注意力机制
        self.attention = SpatialAttention()
        
        # 改进3: 特征金字塔增强
        self.fpn_enhancement = FPNEnhancement()
        
        # 更新分类器和掩码预测器
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes)
            
        # 改进4: 深度可分离卷积
        self.depth_sep_conv = DepthSeparableConv(
            in_channels=256,
            out_channels=256
        )
        
        # 改进5: 边界优化模块
        self.boundary_optimization = BoundaryOptimization()

    def forward(self, images, targets=None):
        if self.training:
            return self._forward_train(images, targets)
        return self._forward_test(images)
    
    def _forward_train(self, images, targets):
        features = self.model.backbone(images)
        features = self.fpn_enhancement(features)
        attention_maps = self.attention(features)
        enhanced_features = {k: v * attention_maps[i] for i, (k, v) in enumerate(features.items())}
        refined_features = {k: self.depth_sep_conv(v) for k, v in enhanced_features.items()}
        return self.model(images, targets)
    
    def _forward_test(self, images):
        return self.model(images)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, features):
        # 计算空间注意力图
        attention_maps = []
        for feat in features.values():
            avg_pool = torch.mean(feat, dim=1, keepdim=True)
            max_pool, _ = torch.max(feat, dim=1, keepdim=True)
            concat = torch.cat([avg_pool, max_pool], dim=1)
            attention_map = torch.sigmoid(self.conv(concat))
            attention_maps.append(attention_map)
        return attention_maps

class FPNEnhancement(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleDict({
            '0': nn.Conv2d(256, 256, kernel_size=1),
            '1': nn.Conv2d(256, 256, kernel_size=1),
            '2': nn.Conv2d(256, 256, kernel_size=1),
            '3': nn.Conv2d(256, 256, kernel_size=1)
        })
        
    def forward(self, features):
        enhanced = {}
        # 只处理FPN输出的特征图
        for k, v in features.items():
            if k in self.lateral_convs:
                enhanced[k] = self.lateral_convs[k](v)
            else:
                enhanced[k] = v
        return enhanced

class DepthSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 
                                  kernel_size=3, padding=1, 
                                  groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BoundaryOptimization(nn.Module):
    def __init__(self):
        super().__init__()
        self.refinement = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1)
        )
    
    def forward(self, proposals):
        refined_proposals = []
        for proposal in proposals:
            # 将边界框转换为掩码形式
            mask = self._box_to_mask(proposal)
            # 优化边界
            refined_mask = self.refinement(mask)
            # 转回边界框形式
            refined_box = self._mask_to_box(refined_mask)
            refined_proposals.append(refined_box)
        return refined_proposals