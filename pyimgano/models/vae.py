import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing

from pyimgano.datasets import ImagePathDataset


class ConvVAE(nn.Module):
    """
    改进版卷积变分自编码器
    - 使用GroupNorm替代BatchNorm
    - 动态计算flatten维度
    - 去除最后的Sigmoid激活
    """

    def __init__(self, input_channels=3, latent_dim=128, input_size=(256, 256)):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # 编码器
        self.encoder_conv = nn.Sequential(
            # 256x256x3 -> 128x128x32
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),  # 使用GroupNorm
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128x32 -> 64x64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64x64 -> 32x32x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32x128 -> 16x16x256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16x256 -> 8x8x512
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 改进6: 动态计算flatten维度
        self.flatten_dim = self._calculate_flatten_dim(input_channels)

        # VAE的关键：输出均值和对数方差
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder_conv = nn.Sequential(
            # 8x8x512 -> 16x16x256
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16x256 -> 32x32x128
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32x128 -> 64x64x64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64x64 -> 128x128x32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128x32 -> 256x256x3
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 改进1: 删除Sigmoid，使用线性输出
        )

    def _calculate_flatten_dim(self, input_channels):
        """动态计算encoder输出的flatten维度"""
        dummy_input = torch.zeros(1, input_channels, *self.input_size)
        with torch.no_grad():
            h = self.encoder_conv(dummy_input)
        return h.numel() // h.size(0)

    def encode(self, x):
        """编码器：返回分布的均值和对数方差"""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码器：从潜在变量重构图像"""
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 512, 8, 8)
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z


class VAELoss(nn.Module):
    """
    VAE损失函数：重构损失 + KL散度
    支持动态beta值（KL annealing）
    """

    def __init__(self, beta=1.0):
        super(VAELoss, self).__init__()
        self.beta = beta

    def forward(self, reconstructed, original, mu, logvar):
        # 重构损失：使用MSE（因为删除了Sigmoid）
        recon_loss = F.mse_loss(reconstructed, original, reduction='none')
        recon_loss = torch.sum(recon_loss, dim=[1, 2, 3])

        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # 总损失
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss


class VAEAnomalyDetector:
    """
    改进版VAE异常检测器
    - 支持KL annealing
    - 优化的阈值计算
    - 简化的异常分数组件
    - 动态num_workers设置
    """

    def __init__(self,
                 input_size=(256, 256),
                 latent_dim=128,
                 batch_size=32,
                 learning_rate=0.001,
                 beta=1.0,
                 kl_warmup_epochs=10,  # 改进3: KL annealing
                 device=None):

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_max = beta
        self.kl_warmup_epochs = kl_warmup_epochs

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"使用设备: {self.device}")

        # 创建模型
        self.model = ConvVAE(input_channels=3, latent_dim=latent_dim,
                             input_size=input_size).to(self.device)
        self.loss_fn = VAELoss(beta=0)  # 初始beta为0

        # 改进1: 数据转换，归一化到[-1, 1]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        # 训练历史
        self.train_losses = []
        self.is_trained = False

        # 改进8: 动态设置num_workers
        self.num_workers = self._get_optimal_num_workers()

    def _get_optimal_num_workers(self):
        """获取最优的num_workers数量"""
        if os.name == 'nt':  # Windows
            return 0
        else:
            return min(4, multiprocessing.cpu_count() // 2)

    def compute_anomaly_score(self, original, reconstructed, mu, logvar):
        """
        改进5: 简化的异常分数计算
        只使用重构误差和KL散度
        """
        batch_size = original.size(0)

        # 1. 重构误差
        recon_error = F.mse_loss(reconstructed, original, reduction='none')
        recon_error = torch.sum(recon_error.view(batch_size, -1), dim=1)

        # 2. KL散度
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # 标准化后加权（使用预先计算的统计信息）
        if hasattr(self, 'component_stats'):
            # Z-score标准化
            recon_z = (recon_error - self.component_stats['recon_error']['mean']) / \
                      (self.component_stats['recon_error']['std'] + 1e-8)
            kl_z = (kl_div - self.component_stats['kl_div']['mean']) / \
                   (self.component_stats['kl_div']['std'] + 1e-8)

            # 组合异常分数
            anomaly_score = recon_z + 0.3 * kl_z
        else:
            # 训练期间的简单组合
            anomaly_score = recon_error + 0.1 * kl_div

        return anomaly_score, {
            'recon_error': recon_error,
            'kl_div': kl_div
        }

    def train(self, data_folder, epochs=50, validation_split=0.1):
        """训练VAE"""
        print(f"开始训练 VAE 异常检测器...")
        print(f"数据目录: {data_folder}")
        print(f"训练轮数: {epochs}")
        print(f"KL warmup epochs: {self.kl_warmup_epochs}")

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

        # 改进8: 使用动态num_workers和条件pin_memory
        pin_memory = self.device.type == 'cuda'
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers,
                                  pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers,
                                pin_memory=pin_memory)

        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         patience=5, factor=0.5)

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 改进3: KL annealing
            if epoch < self.kl_warmup_epochs:
                self.loss_fn.beta = self.beta_max * epoch / self.kl_warmup_epochs
            else:
                self.loss_fn.beta = self.beta_max

            # 训练阶段
            self.model.train()
            train_total_loss = 0
            train_recon_loss = 0
            train_kl_loss = 0

            for batch_idx, (images, _) in enumerate(tqdm(train_loader,
                                                         desc=f"Epoch {epoch + 1}/{epochs}")):
                images = images.to(self.device)

                # 前向传播
                reconstructed, mu, logvar, z = self.model(images)

                # 计算损失
                total_loss, recon_loss, kl_loss = self.loss_fn(reconstructed, images, mu, logvar)
                loss = torch.mean(total_loss)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 改进7: 梯度裁剪阈值调整到5.0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                optimizer.step()

                train_total_loss += loss.item()
                train_recon_loss += torch.mean(recon_loss).item()
                train_kl_loss += torch.mean(kl_loss).item()

            avg_train_loss = train_total_loss / len(train_loader)
            avg_train_recon = train_recon_loss / len(train_loader)
            avg_train_kl = train_kl_loss / len(train_loader)

            # 验证阶段
            self.model.eval()
            val_total_loss = 0
            val_recon_loss = 0
            val_kl_loss = 0

            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(self.device)
                    reconstructed, mu, logvar, z = self.model(images)
                    total_loss, recon_loss, kl_loss = self.loss_fn(reconstructed, images, mu, logvar)

                    val_total_loss += torch.mean(total_loss).item()
                    val_recon_loss += torch.mean(recon_loss).item()
                    val_kl_loss += torch.mean(kl_loss).item()

            avg_val_loss = val_total_loss / len(val_loader)
            avg_val_recon = val_recon_loss / len(val_loader)
            avg_val_kl = val_kl_loss / len(val_loader)

            # 学习率调整
            scheduler.step(avg_val_loss)

            # 记录损失
            self.train_losses.append({
                'epoch': epoch + 1,
                'beta': self.loss_fn.beta,
                'train_loss': avg_train_loss,
                'train_recon': avg_train_recon,
                'train_kl': avg_train_kl,
                'val_loss': avg_val_loss,
                'val_recon': avg_val_recon,
                'val_kl': avg_val_kl
            })

            print(f"Epoch {epoch + 1}: β={self.loss_fn.beta:.3f}, "
                  f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}), "
                  f"Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f})")

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_checkpoint('best_vae.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("早停：验证损失不再改善")
                    break

        # 改进4: 使用验证集计算阈值
        self._calculate_thresholds(val_loader)

        self.is_trained = True
        print("训练完成！")

        return self

    def _calculate_thresholds(self, data_loader):
        """
        改进4: 优化的阈值计算
        - 使用验证集
        - Z-score方法
        - 采样计算
        """
        print("计算异常分数阈值...")

        self.model.eval()
        all_scores = []
        all_components = {
            'recon_error': [],
            'kl_div': []
        }

        # 采样计算（最多5000个样本）
        max_samples = 5000
        samples_processed = 0

        with torch.no_grad():
            for images, _ in tqdm(data_loader, desc="计算异常分数分布"):
                images = images.to(self.device)
                reconstructed, mu, logvar, z = self.model(images)

                anomaly_scores, components = self.compute_anomaly_score(
                    images, reconstructed, mu, logvar
                )

                all_scores.extend(anomaly_scores.cpu().numpy())
                for key, value in components.items():
                    all_components[key].extend(value.cpu().numpy())

                samples_processed += images.size(0)
                if samples_processed >= max_samples:
                    break

        # 计算组件统计信息（用于标准化）
        self.component_stats = {}
        for key, values in all_components.items():
            values = np.array(values)
            self.component_stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

        # 重新计算标准化后的异常分数
        all_scores_normalized = []
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                reconstructed, mu, logvar, z = self.model(images)
                anomaly_scores, _ = self.compute_anomaly_score(
                    images, reconstructed, mu, logvar
                )
                all_scores_normalized.extend(anomaly_scores.cpu().numpy())

                if len(all_scores_normalized) >= max_samples:
                    break

        all_scores = np.array(all_scores_normalized[:max_samples])

        # 使用Z-score方法设置阈值
        score_mean = np.mean(all_scores)
        score_std = np.std(all_scores)

        self.threshold = score_mean + 3 * score_std
        self.threshold_high = score_mean + 4 * score_std
        self.threshold_medium = score_mean + 2 * score_std

        # 保存统计信息
        self.score_stats = {
            'mean': float(score_mean),
            'std': float(score_std),
            'min': float(np.min(all_scores)),
            'max': float(np.max(all_scores))
        }

        print(f"\n异常分数统计:")
        print(f"  均值 (μ): {score_mean:.4f}")
        print(f"  标准差 (σ): {score_std:.4f}")
        print(f"  异常检测阈值 (μ+3σ): {self.threshold:.4f}")
        print(f"  高异常阈值 (μ+4σ): {self.threshold_high:.4f}")

    def predict(self, image_path):
        """预测单张图片是否异常"""
        if not self.is_trained:
            raise ValueError("模型未训练！")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 读取并预处理图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()

        # 转换为tensor
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 预测
        self.model.eval()
        with torch.no_grad():
            reconstructed, mu, logvar, z = self.model(image_tensor)
            anomaly_score, components = self.compute_anomaly_score(
                image_tensor, reconstructed, mu, logvar
            )

        # 转换为numpy
        anomaly_score = anomaly_score.cpu().numpy()[0]
        components = {k: v.cpu().numpy()[0] for k, v in components.items()}

        # 判断异常等级
        if anomaly_score > self.threshold_high:
            anomaly_level = "高异常"
            is_normal = False
        elif anomaly_score > self.threshold:
            anomaly_level = "中异常"
            is_normal = False
        elif anomaly_score > self.threshold_medium:
            anomaly_level = "轻微异常"
            is_normal = False
        else:
            anomaly_level = "正常"
            is_normal = True

        # 计算置信度（基于Z-score）
        z_score = (anomaly_score - self.score_stats['mean']) / self.score_stats['std']
        confidence = 1 / (1 + np.exp(-np.clip(z_score - 3, -10, 10)))

        result = {
            'image': os.path.basename(image_path),
            'is_normal': is_normal,
            'prediction': '正常' if is_normal else '异常',
            'anomaly_level': anomaly_level,
            'anomaly_score': float(anomaly_score),
            'z_score': float(z_score),
            'components': {k: float(v) for k, v in components.items()},
            'threshold': float(self.threshold),
            'confidence': float(confidence)
        }

        # 保存重构图像用于可视化
        self.last_reconstruction = {
            'original': original_image,
            'reconstructed': self._tensor_to_image(reconstructed[0]),
            'mu': mu.cpu().numpy()[0],
            'logvar': logvar.cpu().numpy()[0],
            'z': z.cpu().numpy()[0]
        }

        return result

    def _tensor_to_image(self, tensor):
        """将tensor转换为numpy图像（反标准化）"""
        # 反标准化：从[-1, 1]到[0, 1]
        tensor = tensor * 0.5 + 0.5
        image = tensor.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image

    def visualize_results(self, save_path=None):
        """可视化VAE结果"""
        if not hasattr(self, 'last_reconstruction'):
            print("没有可视化的数据，请先运行predict")
            return

        fig = plt.figure(figsize=(20, 5))

        # 1. 原图
        ax1 = plt.subplot(1, 5, 1)
        ax1.imshow(self.last_reconstruction['original'])
        ax1.set_title('原始图像')
        ax1.axis('off')

        # 2. 重构图
        ax2 = plt.subplot(1, 5, 2)
        ax2.imshow(self.last_reconstruction['reconstructed'])
        ax2.set_title('重构图像')
        ax2.axis('off')

        # 3. 误差热图
        ax3 = plt.subplot(1, 5, 3)
        original = self.last_reconstruction['original'].astype(float) / 255.0
        reconstructed = cv2.resize(self.last_reconstruction['reconstructed'],
                                   (original.shape[1], original.shape[0])).astype(float) / 255.0
        error = np.abs(original - reconstructed).mean(axis=2)
        im = ax3.imshow(error, cmap='hot')
        ax3.set_title('重构误差热图')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3)

        # 4. 潜在变量分布
        ax4 = plt.subplot(1, 5, 4)
        z = self.last_reconstruction['z']
        ax4.hist(z, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', label='N(0,1)中心')
        ax4.set_title(f'潜在变量分布\n(dim={len(z)})')
        ax4.set_xlabel('值')
        ax4.set_ylabel('频率')
        ax4.legend()

        # 5. 潜在空间统计
        ax5 = plt.subplot(1, 5, 5)
        mu = self.last_reconstruction['mu']
        logvar = self.last_reconstruction['logvar']
        std = np.exp(0.5 * logvar)

        # 显示前10个维度的均值和标准差
        n_show = min(10, len(mu))
        x = np.arange(n_show)
        width = 0.35

        ax5.bar(x - width / 2, mu[:n_show], width, label='μ (均值)', alpha=0.8)
        ax5.bar(x + width / 2, std[:n_show], width, label='σ (标准差)', alpha=0.8)
        ax5.set_xlabel('潜在维度')
        ax5.set_ylabel('值')
        ax5.set_title('潜在空间参数\n(前10维)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        else:
            plt.show()

    def generate_samples(self, n_samples=16, z=None):
        """
        改进10: 修复生成样本的图排版
        """
        self.model.eval()

        with torch.no_grad():
            if z is None:
                # 从标准正态分布采样
                z = torch.randn(n_samples, self.latent_dim).to(self.device)

            # 解码生成图像
            generated = self.model.decode(z)

        # 可视化生成的样本
        n_show = min(n_samples, 16)  # 最多显示16个
        n_cols = 4
        n_rows = (n_show + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.5 * n_rows))
        axes = axes.ravel() if n_rows > 1 else [axes]

        for i in range(n_show):
            img = self._tensor_to_image(generated[i])
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'生成样本 {i + 1}')

        # 隐藏多余的子图
        for i in range(n_show, len(axes)):
            axes[i].axis('off')

        plt.suptitle('VAE生成的样本', fontsize=16)
        plt.tight_layout()
        plt.show()

        return generated

    def plot_training_history(self):
        """绘制详细的训练历史（包含beta变化）"""
        if not self.train_losses:
            print("没有训练历史")
            return

        epochs = [loss['epoch'] for loss in self.train_losses]
        betas = [loss['beta'] for loss in self.train_losses]
        train_losses = [loss['train_loss'] for loss in self.train_losses]
        val_losses = [loss['val_loss'] for loss in self.train_losses]
        train_recons = [loss['train_recon'] for loss in self.train_losses]
        val_recons = [loss['val_recon'] for loss in self.train_losses]
        train_kls = [loss['train_kl'] for loss in self.train_losses]
        val_kls = [loss['val_kl'] for loss in self.train_losses]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Beta变化
        axes[0, 0].plot(epochs, betas, label='β值', marker='o', color='purple')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('β')
        axes[0, 0].set_title('KL Annealing (β变化)')
        axes[0, 0].grid(True, alpha=0.3)

        # 总损失
        axes[0, 1].plot(epochs, train_losses, label='训练损失', marker='o')
        axes[0, 1].plot(epochs, val_losses, label='验证损失', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('总损失')
        axes[0, 1].set_title('总损失变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 重构损失
        axes[1, 0].plot(epochs, train_recons, label='训练重构损失', marker='o')
        axes[1, 0].plot(epochs, val_recons, label='验证重构损失', marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('重构损失')
        axes[1, 0].set_title('重构损失变化')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # KL散度
        axes[1, 1].plot(epochs, train_kls, label='训练KL散度', marker='o')
        axes[1, 1].plot(epochs, val_kls, label='验证KL散度', marker='s')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('KL散度')
        axes[1, 1].set_title('KL散度变化')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save(self, filepath):
        """保存完整模型"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'latent_dim': self.latent_dim,
            'beta_max': self.beta_max,
            'kl_warmup_epochs': self.kl_warmup_epochs,
            'threshold': self.threshold,
            'threshold_high': self.threshold_high,
            'threshold_medium': self.threshold_medium,
            'score_stats': self.score_stats,
            'component_stats': self.component_stats,
            'train_losses': self.train_losses,
            'is_trained': self.is_trained
        }
        torch.save(save_dict, filepath)
        print(f"VAE 模型已保存到: {filepath}")

    def load(self, filepath):
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # 重新创建模型（确保input_size匹配）
        self.input_size = checkpoint['input_size']
        self.latent_dim = checkpoint['latent_dim']
        self.model = ConvVAE(input_channels=3, latent_dim=self.latent_dim,
                             input_size=self.input_size).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.beta_max = checkpoint.get('beta_max', 1.0)
        self.kl_warmup_epochs = checkpoint.get('kl_warmup_epochs', 10)
        self.threshold = checkpoint['threshold']
        self.threshold_high = checkpoint['threshold_high']
        self.threshold_medium = checkpoint['threshold_medium']
        self.score_stats = checkpoint['score_stats']
        self.component_stats = checkpoint.get('component_stats', {})
        self.train_losses = checkpoint.get('train_losses', [])
        self.is_trained = checkpoint['is_trained']

        print(f"模型已从 {filepath} 加载")

    def save_checkpoint(self, filepath):
        """保存训练检查点"""
        torch.save(self.model.state_dict(), filepath)


# 使用示例
if __name__ == "__main__":
    # Windows多进程兼容性
    if os.name == 'nt':
        multiprocessing.freeze_support()

    # 创建改进版VAE检测器
    detector = VAEAnomalyDetector(
        input_size=(256, 256),
        latent_dim=128,
        batch_size=32,
        learning_rate=0.001,
        beta=1.0,
        kl_warmup_epochs=10  # KL annealing
    )

    try:
        # 训练模型
        train_folder = "/data/temp7/程序正常"
        detector.train(train_folder, epochs=50, validation_split=0.1)

        # 保存模型
        detector.save("vae_anomaly_detector_improved.pth")

        # 绘制训练历史
        detector.plot_training_history()

        # 测试单张图片
        # test_image = "test.jpg"
        # if os.path.exists(test_image):
        #     result = detector.predict(test_image)
        #     print(f"\n单张图片检测结果:")
        #     print(f"  图片: {result['image']}")
        #     print(f"  预测: {result['prediction']}")
        #     print(f"  异常等级: {result['anomaly_level']}")
        #     print(f"  异常分数: {result['anomaly_score']:.4f}")
        #     print(f"  Z-score: {result['z_score']:.2f}")
        #     print(f"  置信度: {result['confidence']:.2f}")
        #     print(f"\n  分数组成:")
        #     for key, value in result['components'].items():
        #         print(f"    {key}: {value:.4f}")
        #
        #     # 可视化结果
        #     detector.visualize_results(save_path="vae_result_improved.png")
        #
        # # 生成新样本
        # print("\n生成新样本...")
        # detector.generate_samples(n_samples=16)

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()
