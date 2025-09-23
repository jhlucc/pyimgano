import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pytorch_msssim import ssim, SSIM
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available, using fallback method for latent distance")
import multiprocessing
from sklearn.decomposition import PCA as sklearn_PCA

from pyimgano.datasets import ImagePathDataset
from .registry import register_model


class ResNetUNetAE(nn.Module):
    """
    轻量化U-Net架构的自编码器
    - 使用ResNet18作为encoder backbone
    - 使用PixelShuffle上采样避免棋盘效应
    - GroupNorm替代BatchNorm
    """

    def __init__(self, latent_dim=64, pretrained=True):
        super(ResNetUNetAE, self).__init__()

        # 加载预训练的ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # Encoder: 使用ResNet18的前几层
        # Layer1: 64 channels, /4
        # Layer2: 128 channels, /8
        # Layer3: 256 channels, /16
        # Layer4: 512 channels, /32
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )  # 64 channels, /4

        self.encoder2 = resnet.layer1  # 64 channels, /4
        self.encoder3 = resnet.layer2  # 128 channels, /8
        self.encoder4 = resnet.layer3  # 256 channels, /16
        self.encoder5 = resnet.layer4  # 512 channels, /32

        # 替换所有BatchNorm为GroupNorm
        self._replace_bn_with_gn(self.encoder1)
        self._replace_bn_with_gn(self.encoder2)
        self._replace_bn_with_gn(self.encoder3)
        self._replace_bn_with_gn(self.encoder4)
        self._replace_bn_with_gn(self.encoder5)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_dim, 1),
            nn.GroupNorm(min(16, latent_dim // 4), latent_dim),  # 适应不同的latent_dim
            nn.ReLU(inplace=True)
        )

        # Decoder with skip connections (U-Net style)
        # 使用PixelShuffle代替转置卷积
        self.decoder5 = nn.Sequential(
            nn.Conv2d(latent_dim, 256 * 4, 1),  # 64 -> 1024
            nn.PixelShuffle(2),  # 1024 -> 256 (channels/4, size*2)
            nn.Conv2d(256, 256, 3, padding=1),  # Fixed: 256 input channels
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256 * 4, 1),  # 512 → 1024
            nn.PixelShuffle(2),  # 1024 → 256
            nn.Conv2d(256, 128, 3, padding=1),  # Fixed: 256 input channels
            nn.GroupNorm(8, 128),
            nn.ReLU(True)
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128 * 4, 1),  # 256 → 512
            nn.PixelShuffle(2),  # 512 → 128
            nn.Conv2d(128, 64, 3, padding=1),  # Fixed: 128 input channels
            nn.GroupNorm(8, 64),
            nn.ReLU(True)
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64 * 4, 1),  # 128 → 256
            nn.PixelShuffle(2),  # 256 → 64
            nn.Conv2d(64, 32, 3, padding=1),  # Fixed: 64 input channels
            nn.GroupNorm(4, 32),
            nn.ReLU(True)
        )

        # Fixed decoder1 - no skip connection from e1 due to resolution mismatch
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1),  # No skip connection, just d2
            nn.PixelShuffle(2),  # 32*4 -> 32 channels, size*2 (to full resolution)
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def _replace_bn_with_gn(self, module):
        """递归替换BatchNorm为GroupNorm"""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                # 更合理的分组策略
                num_channels = child.num_features
                if num_channels >= 128:
                    num_groups = 32
                elif num_channels >= 64:
                    num_groups = 16
                else:
                    num_groups = min(8, num_channels // 8)
                setattr(module, name, nn.GroupNorm(num_groups, num_channels))
            else:
                self._replace_bn_with_gn(child)

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.encoder1(x)  # 64, H/4, W/4
        e2 = self.encoder2(e1)  # 64, H/4, W/4
        e3 = self.encoder3(e2)  # 128, H/8, W/8
        e4 = self.encoder4(e3)  # 256, H/16, W/16
        e5 = self.encoder5(e4)  # 512, H/32, W/32

        # Bottleneck
        latent = self.bottleneck(e5)  # latent_dim, H/32, W/32

        # Decoder with skip connections
        d5 = self.decoder5(latent)  # 256, H/16, W/16
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))  # 128, H/8, W/8
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))  # 64, H/4, W/4
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))  # 32, H/2, W/2
        d1 = self.decoder1(d2)  # 3, H, W - No concatenation with e1

        return d1, latent

    def get_latent_features(self, x):
        """获取潜在特征用于异常定位"""
        with torch.no_grad():
            e1 = self.encoder1(x)
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)
            e5 = self.encoder5(e4)
            latent = self.bottleneck(e5)
        return latent

class ComposedLoss(nn.Module):
    """L1 + SSIM 复合损失"""

    def __init__(self, alpha=0.85):
        super(ComposedLoss, self).__init__()
        self.alpha = alpha
        self.ssim = SSIM(data_range=1.0, channel=3)

    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss


@register_model(
    "ae_resnet_unet",
    tags=("vision", "deep", "autoencoder"),
    metadata={"description": "基于 ResNet-UNet 的重建式异常检测器"},
)
class OptimizedAEDetector:
    """
    优化版自编码器异常检测器
    - ResNet18 backbone
    - L1+SSIM损失
    - 动态阈值
    - 支持ONNX导出
    """

    def __init__(self,
                 input_size=(256, 256),
                 latent_dim=64,
                 batch_size=32,
                 learning_rate=0.0005,
                 device=None):

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"使用设备: {self.device}")

        # 创建模型
        self.model = ResNetUNetAE(latent_dim=latent_dim).to(self.device)
        self.loss_fn = ComposedLoss(alpha=0.85)

        # 数据转换 - 添加ImageNet标准化
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet预训练分布
        ])

        # 训练历史
        self.train_losses = []
        self.val_error_stds = []  # 监控误差分布的标准差
        self.is_trained = False

        # 动态阈值参数
        self.error_mean = None
        self.error_std = None
        self.ema_alpha = 0.01  # EMA更新系数
        self.error_history = deque(maxlen=5000)  # 限制最大长度
        self.last_threshold_update = datetime.now()
        self.update_interval = timedelta(minutes=30)  # 半小时更新一次

        # FAISS索引用于加速latent distance计算
        self.faiss_index = None
        self.faiss_pca = None
        self.latent_dim_compressed = 64  # PCA压缩后的维度

        # 保存训练集的latent features用于异常定位
        self.train_latent_features = None

    def compute_reconstruction_error(self, original, reconstructed):
        """计算重构误差（使用avg_pool替代unfold）"""
        batch_size = original.size(0)

        # 像素级MSE
        pixel_errors = torch.mean((original - reconstructed) ** 2, dim=1)  # [B, H, W]

        # 使用avg_pool2d计算局部误差
        patch_errors = F.avg_pool2d(
            pixel_errors.unsqueeze(1),
            kernel_size=16,
            stride=16
        ).squeeze(1)  # [B, H/16, W/16]

        # 取最大的几个patch误差
        patch_errors_flat = patch_errors.view(batch_size, -1)
        top_k = min(10, patch_errors_flat.size(1))
        local_max_errors, _ = torch.topk(patch_errors_flat, k=top_k, dim=1)
        local_max_error = torch.mean(local_max_errors, dim=1)

        # 全局MSE
        mse = torch.mean(pixel_errors.view(batch_size, -1), dim=1)

        # 组合误差
        total_error = mse + 0.5 * local_max_error

        return total_error, mse, local_max_error

    def train(self, data_folder, epochs=50, validation_split=0.1):
        """训练自编码器"""
        print(f"开始训练优化版 Autoencoder...")
        print(f"数据目录: {data_folder}")
        print(f"训练轮数: {epochs}")

        # 获取所有图像路径
        image_paths = []
        for filename in os.listdir(data_folder):
            if filename.endswith('.jpg'):
                image_paths.append(os.path.join(data_folder, filename))

        if not image_paths:
            raise ValueError(f"在 {data_folder} 中没有找到jpg文件")

        print(f"找到 {len(image_paths)} 张图像")

        # 划分训练集和验证集
        np.random.seed(42)
        np.random.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - validation_split))
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        print(f"训练集: {len(train_paths)} 张")
        print(f"验证集: {len(val_paths)} 张")

        # 创建数据加载器
        train_dataset = ImagePathDataset(train_paths, transform=self.transform)
        val_dataset = ImagePathDataset(val_paths, transform=self.transform)

        # 动态设置num_workers
        num_workers = min(8, multiprocessing.cpu_count())
        if os.name == 'nt':  # Windows
            num_workers = 0  # Windows下多进程可能有问题

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True)

        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         patience=5, factor=0.5)

        # 训练循环
        best_val_loss = float('inf')
        best_error_std = float('inf')
        patience_counter = 0
        std_patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0

            for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
                images = images.to(self.device)

                # 前向传播
                reconstructed, latent = self.model(images)

                # 计算损失
                loss = self.loss_fn(reconstructed, images)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_errors = []

            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(self.device)
                    reconstructed, latent = self.model(images)

                    # 损失
                    loss = self.loss_fn(reconstructed, images)
                    val_loss += loss.item()

                    # 计算误差用于监控分布
                    errors, _, _ = self.compute_reconstruction_error(images, reconstructed)
                    val_errors.extend(errors.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_error_std = np.std(val_errors)

            # 记录
            self.train_losses.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_error_std': val_error_std
            })

            print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, Val Error Std: {val_error_std:.6f}")

            # 学习率调整
            scheduler.step(avg_val_loss)

            # 早停检查 - 同时监控loss和误差分布
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_checkpoint('best_ae_optimized.pth')
            else:
                patience_counter += 1

            # 监控误差分布的标准差 - 保存std最小的模型
            if val_error_std < best_error_std:
                best_error_std = val_error_std
                std_patience_counter = 0
                self.save_checkpoint('best_ae_optimized_std.pth')  # 额外保存
            else:
                std_patience_counter += 1

            # 早停条件
            if patience_counter >= 10 and std_patience_counter >= 5:
                print("早停：验证损失和误差分布都不再改善")
                break

        # 计算阈值并保存latent features
        self._calculate_thresholds_and_features(train_loader)

        self.is_trained = True
        print("训练完成！")

        return self

    def _calculate_thresholds_and_features(self, data_loader):
        """计算动态阈值参数并使用FAISS建立latent索引"""
        print("计算阈值参数和建立FAISS索引...")

        self.model.eval()
        all_errors = []
        all_latent_features = []

        with torch.no_grad():
            for images, _ in tqdm(data_loader, desc="计算误差分布"):
                images = images.to(self.device)
                reconstructed, latent = self.model(images)

                errors, _, _ = self.compute_reconstruction_error(images, reconstructed)
                all_errors.extend(errors.cpu().numpy())

                # 保存latent features
                all_latent_features.append(latent.cpu())

        # 计算误差统计（μ + 3σ方法）
        all_errors = np.array(all_errors)
        self.error_mean = float(np.mean(all_errors))
        self.error_std = float(np.std(all_errors))

        # 设置阈值
        self.threshold = self.error_mean + 3.5 * self.error_std
        self.threshold_high = self.error_mean + 4.5 * self.error_std
        self.threshold_medium = self.error_mean + 2.5 * self.error_std

        # 处理latent features
        self.train_latent_features = torch.cat(all_latent_features, dim=0)

        # 建立FAISS索引
        if FAISS_AVAILABLE:
            self._build_faiss_index()
        else:
            print("FAISS不可用，将使用原始方法计算距离")

        print(f"\n误差统计:")
        print(f"  均值 (μ): {self.error_mean:.6f}")
        print(f"  标准差 (σ): {self.error_std:.6f}")
        print(f"  异常阈值 (μ+3.5σ): {self.threshold:.6f}")
        if self.faiss_index is not None:
            print(f"  FAISS索引已建立，压缩后维度: {self.latent_dim_compressed}")

    def _build_faiss_index(self):
        """建立FAISS索引用于快速最近邻搜索"""
        # 将latent features展平
        n, c, h, w = self.train_latent_features.shape
        features_flat = self.train_latent_features.view(n, -1).numpy()

        # PCA降维到64维
        self.faiss_pca = sklearn_PCA(n_components=min(self.latent_dim_compressed, features_flat.shape[1]))
        features_compressed = self.faiss_pca.fit_transform(features_flat)

        # 创建FAISS L2索引
        dim = features_compressed.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(features_compressed.astype('float32'))

        print(f"FAISS索引建立完成: {n} 个样本, {dim} 维特征")

    def update_threshold_online(self, error_value):
        """在线更新阈值（限制更新频率）"""
        if self.error_mean is not None:
            self.error_history.append(error_value)

            # 检查是否满足更新条件
            current_time = datetime.now()
            time_since_update = current_time - self.last_threshold_update

            if (len(self.error_history) >= 100 and
                    time_since_update >= self.update_interval):
                # 计算最近误差的统计
                recent_errors = list(self.error_history)[-1000:]  # 最近1000个
                recent_mean = np.mean(recent_errors)
                recent_std = np.std(recent_errors)

                # EMA更新
                self.error_mean = (1 - self.ema_alpha) * self.error_mean + self.ema_alpha * recent_mean
                self.error_std = (1 - self.ema_alpha) * self.error_std + self.ema_alpha * recent_std

                # 更新阈值
                self.threshold = self.error_mean + 3.5 * self.error_std
                self.threshold_high = self.error_mean + 4.5 * self.error_std
                self.threshold_medium = self.error_mean + 2.5 * self.error_std

                # 更新时间戳
                self.last_threshold_update = current_time

                print(f"阈值已更新: μ={self.error_mean:.4f}, σ={self.error_std:.4f}")

    def compute_latent_distance_map(self, test_latent):
        """计算latent distance map用于异常定位"""
        if self.train_latent_features is None:
            return None

        # test_latent: [1, C, H, W]
        # train_latent_features: [N, C, H, W]

        b, c, h, w = test_latent.shape
        test_latent_flat = test_latent.view(1, c, -1).permute(0, 2, 1)  # [1, H*W, C]

        # 计算与训练集最近样本的距离
        min_distances = []

        # 分批处理避免显存溢出
        batch_size = 100
        for i in range(0, self.train_latent_features.size(0), batch_size):
            batch_train = self.train_latent_features[i:i + batch_size]
            batch_train_flat = batch_train.view(batch_train.size(0), c, -1).permute(0, 2, 1)  # [B, H*W, C]

            # 计算余弦相似度
            distances = 1 - F.cosine_similarity(
                test_latent_flat.unsqueeze(1),  # [1, 1, H*W, C]
                batch_train_flat.unsqueeze(0),  # [1, B, H*W, C]
                dim=-1
            )  # [1, B, H*W]

            # 取最小距离
            min_dist, _ = torch.min(distances, dim=1)  # [1, H*W]
            min_distances.append(min_dist)

        # 合并所有批次的最小距离
        all_min_distances = torch.cat(min_distances, dim=0)
        final_min_distance, _ = torch.min(all_min_distances.view(-1, h * w), dim=0)

        # 重塑为2D map
        distance_map = final_min_distance.view(h, w)

        # 上采样到原始大小
        distance_map_upsampled = F.interpolate(
            distance_map.unsqueeze(0).unsqueeze(0),
            size=self.input_size,
            mode='bilinear',
            align_corners=False
        ).squeeze()

        return distance_map_upsampled

    def predict(self, image_path):
        """预测单张图片"""
        if not self.is_trained:
            raise ValueError("模型未训练！")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()

        # 转换
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 预测
        self.model.eval()
        with torch.no_grad():
            reconstructed, latent = self.model(image_tensor)
            total_error, mse_error, local_error = self.compute_reconstruction_error(
                image_tensor, reconstructed
            )

            # 计算latent distance map
            distance_map = self.compute_latent_distance_map(latent)

        # 转换为numpy
        total_error = total_error.cpu().numpy()[0]
        mse_error = mse_error.cpu().numpy()[0]
        local_error = local_error.cpu().numpy()[0]

        # 在线更新阈值（如果是正常样本）
        if total_error < self.threshold:
            self.update_threshold_online(total_error)

        # 判断异常等级
        if total_error > self.threshold_high:
            anomaly_level = "高异常"
            is_normal = False
        elif total_error > self.threshold:
            anomaly_level = "中异常"
            is_normal = False
        elif total_error > self.threshold_medium:
            anomaly_level = "轻微异常"
            is_normal = False
        else:
            anomaly_level = "正常"
            is_normal = True

        # 计算置信度
        z_score = (total_error - self.error_mean) / (self.error_std + 1e-8)
        confidence = 1 - (1 / (1 + np.exp(-np.clip(z_score, -10, 10))))

        result = {
            'image': os.path.basename(image_path),
            'is_normal': is_normal,
            'prediction': '正常' if is_normal else '异常',
            'anomaly_level': anomaly_level,
            'reconstruction_error': float(total_error),
            'mse_error': float(mse_error),
            'local_max_error': float(local_error),
            'threshold': float(self.threshold),
            'confidence': float(confidence),
            'z_score': float(z_score),
            'dynamic_threshold': {
                'mean': float(self.error_mean),
                'std': float(self.error_std)
            }
        }

        # 保存可视化数据
        self.last_reconstruction = {
            'original': original_image,
            'reconstructed': self._tensor_to_image(reconstructed[0]),
            'distance_map': distance_map.cpu().numpy() if distance_map is not None else None
        }

        return result

    def export_onnx(self, onnx_path="ae_optimized.onnx", quantize=True):
        """导出ONNX模型并可选择性进行PTQ量化"""
        print("导出ONNX模型...")

        self.model.eval()
        dummy_input = torch.randn(1, 3, *self.input_size).to(self.device)

        # 先导出标准ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output', 'latent'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
                'latent': {0: 'batch_size'}
            }
        )

        print(f"ONNX模型已保存到: {onnx_path}")

        if quantize:
            # PTQ量化
            try:
                print("\n尝试PTQ量化...")
                import torch.quantization as tq

                # 准备量化
                self.model.cpu()
                self.model.qconfig = tq.get_default_qconfig('fbgemm')

                # 准备模型
                model_prepared = tq.prepare(self.model, inplace=False)

                # 使用一些样本进行校准
                calibration_data = torch.randn(10, 3, *self.input_size)
                with torch.no_grad():
                    for i in range(10):
                        model_prepared(calibration_data[i:i + 1])

                # 转换为量化模型
                model_quantized = tq.convert(model_prepared, inplace=False)

                # 导出量化模型
                quantized_path = onnx_path.replace('.onnx', '_quantized.onnx')
                torch.onnx.export(
                    model_quantized,
                    calibration_data[0:1],
                    quantized_path,
                    export_params=True,
                    opset_version=13,  # 量化需要更高版本
                    do_constant_folding=True
                )

                print(f"量化模型已保存到: {quantized_path}")

            except Exception as e:
                print(f"PTQ量化失败: {e}")
                print("\n手动量化建议:")
                print("1. pip install onnxruntime-tools")
                print(
                    "2. python -m onnxruntime.quantization.preprocess --input ae_optimized.onnx --output ae_optimized_prep.onnx")
                print(
                    "3. python -m onnxruntime.quantization.quantize --input ae_optimized_prep.onnx --output ae_optimized_int8.onnx")
                print("   --quant_format QDQ --per_channel --activation_type QUInt8 --weight_type QInt8")

    def visualize_anomaly_localization(self, save_path=None):
        """可视化异常定位结果"""
        if not hasattr(self, 'last_reconstruction'):
            print("没有可视化的数据")
            return

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 原图
        axes[0].imshow(self.last_reconstruction['original'])
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        # 重构图
        axes[1].imshow(self.last_reconstruction['reconstructed'])
        axes[1].set_title('重构图像')
        axes[1].axis('off')

        # 像素误差
        original = self.last_reconstruction['original'].astype(float)
        reconstructed = cv2.resize(self.last_reconstruction['reconstructed'],
                                   (original.shape[1], original.shape[0]))
        pixel_error = np.abs(original - reconstructed).mean(axis=2)

        im2 = axes[2].imshow(pixel_error, cmap='hot')
        axes[2].set_title('像素误差热图')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])

        # Latent distance map
        if self.last_reconstruction['distance_map'] is not None:
            im3 = axes[3].imshow(self.last_reconstruction['distance_map'], cmap='hot')
            axes[3].set_title('潜在空间距离图\n(异常定位)')
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3])
        else:
            axes[3].text(0.5, 0.5, 'No distance map', ha='center', va='center')
            axes[3].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        else:
            plt.show()

    def _tensor_to_image(self, tensor):
        """将tensor转换为numpy图像（包括反标准化）"""
        # 反标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor.cpu() * std + mean

        # 转换为numpy
        image = tensor.numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image

    def save(self, filepath):
        """保存模型和FAISS索引"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'latent_dim': self.latent_dim,
            'error_mean': self.error_mean,
            'error_std': self.error_std,
            'threshold': self.threshold,
            'threshold_high': self.threshold_high,
            'threshold_medium': self.threshold_medium,
            'train_losses': self.train_losses,
            'is_trained': self.is_trained,
            'last_threshold_update': self.last_threshold_update
        }

        # 保存FAISS相关
        if FAISS_AVAILABLE and self.faiss_index is not None:
            # 保存FAISS索引
            faiss_path = filepath.replace('.pth', '_faiss.index')
            faiss.write_index(self.faiss_index, faiss_path)
            save_dict['faiss_index_path'] = faiss_path

            # 保存PCA模型
            pca_path = filepath.replace('.pth', '_pca.pkl')
            joblib.dump(self.faiss_pca, pca_path)
            save_dict['pca_path'] = pca_path

            print(f"FAISS索引已保存到: {faiss_path}")
            print(f"PCA模型已保存到: {pca_path}")
        else:
            # 保存原始latent features
            save_dict['train_latent_features'] = self.train_latent_features

        torch.save(save_dict, filepath)
        print(f"优化版AE模型已保存到: {filepath}")

    def load(self, filepath):
        """加载模型和FAISS索引"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.input_size = checkpoint['input_size']
        self.latent_dim = checkpoint['latent_dim']
        self.error_mean = checkpoint['error_mean']
        self.error_std = checkpoint['error_std']
        self.threshold = checkpoint['threshold']
        self.threshold_high = checkpoint['threshold_high']
        self.threshold_medium = checkpoint['threshold_medium']
        self.train_losses = checkpoint.get('train_losses', [])
        self.is_trained = checkpoint['is_trained']
        self.last_threshold_update = checkpoint.get('last_threshold_update', datetime.now())

        # 加载FAISS索引
        if FAISS_AVAILABLE and 'faiss_index_path' in checkpoint:
            try:
                self.faiss_index = faiss.read_index(checkpoint['faiss_index_path'])
                self.faiss_pca = joblib.load(checkpoint['pca_path'])
                print(f"FAISS索引已加载")
            except Exception as e:
                print(f"加载FAISS索引失败: {e}")
                self.train_latent_features = checkpoint.get('train_latent_features')
        else:
            self.train_latent_features = checkpoint.get('train_latent_features')

        print(f"模型已从 {filepath} 加载")

    def save_checkpoint(self, filepath):
        """保存训练检查点"""
        torch.save(self.model.state_dict(), filepath)


# 使用示例
if __name__ == "__main__":
    # Windows多进程兼容性
    if os.name == 'nt':
        multiprocessing.freeze_support()

    # 创建优化版检测器
    detector = OptimizedAEDetector(
        input_size=(256, 256),
        latent_dim=64,  # 更小的潜在维度
        batch_size=32,
        learning_rate=0.001
    )

    try:
        # 训练
        train_folder = "/data/temp7/程序正常"
        detector.train(train_folder, epochs=50, validation_split=0.1)

        # 保存模型
        detector.save("ae_resnet_optimized.pth")

        # 导出ONNX（带量化）
        detector.export_onnx("ae_resnet_optimized.onnx", quantize=True)

        # 测试
        test_image = "test.jpg"
        if os.path.exists(test_image):
            result = detector.predict(test_image)
            print(f"\n检测结果:")
            print(f"  预测: {result['prediction']}")
            print(f"  异常等级: {result['anomaly_level']}")
            print(f"  重构误差: {result['reconstruction_error']:.4f}")
            print(f"  动态阈值: μ={result['dynamic_threshold']['mean']:.4f}, "
                  f"σ={result['dynamic_threshold']['std']:.4f}")
            print(f"  Z-score: {result['z_score']:.2f}")

            # 可视化异常定位
            detector.visualize_anomaly_localization("anomaly_localization.png")

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()
