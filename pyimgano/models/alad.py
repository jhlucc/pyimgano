# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Optional
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .baseCv import BaseVisionDeepDetector


def get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvEncoder(nn.Module):
    """将 3x224x224 图像编码为潜变量 z。
    结构：一串 stride=2 的卷积降采样到 7x7，再 GAP 到向量，最后线性到 latent_dim。
    """
    def __init__(self, latent_dim: int, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1), nn.BatchNorm2d(base_ch * 8), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1), nn.BatchNorm2d(base_ch * 8), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_ch * 8, latent_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.gap(h).flatten(1)  # [B, C]
        z = self.fc(h)
        return z


class ConvDecoder(nn.Module):
    """从潜变量 z 解码回 3x224x224 图像。
    结构：FC -> reshape(512,7,7) -> 反卷积逐步上采样到 224。
    最后一层不加激活（依赖判别器学习到合适的范围；输入通常是已标准化的张量）。
    """

    def __init__(self, latent_dim: int, out_ch: int = 3, base_ch: int = 64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_ch * 8 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 4, 2, 1), nn.BatchNorm2d(base_ch * 8), nn.ReLU(inplace=True),
            # 7->14
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
            # 14->28
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            # 28->56
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            # 56->112
            nn.ConvTranspose2d(base_ch, out_ch, 4, 2, 1),  # 112->224
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), -1, 7, 7)
        x_hat = self.deconv(h)
        return x_hat

class DiscXX(nn.Module):
    """在图像对 (x, x') 上判别真假；输入按通道拼接为 6xHxW。"""

    def __init__(self, in_ch_pair: int = 6, base_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch_pair, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), nn.BatchNorm2d(base_ch * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1), nn.BatchNorm2d(base_ch * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(base_ch * 8, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, x2], dim=1)  # [B, 6, H, W]
        h = self.net(h)
        out = self.fc(h)
        return out  # [B, 1]

class ImgFeat(nn.Module):
    """图像特征提取器，用于 D_xz。
    产出一个中等维度的向量（默认 256）。
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 64, feat_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),  # 224->112
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),  # 112->56
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),  # 56->28
            nn.Conv2d(base_ch * 4, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),  # 28->14
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(base_ch * 4, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.proj(h)

class MLPDisc(nn.Module):
    """多层感知机判别器（用于 D_xz 和 D_zz）。"""
    def __init__(self, in_dim: int, hidden: List[int], act: str = 'tanh', spectral_norm: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            lin = nn.Linear(last, h)
            if spectral_norm:
                lin = nn.utils.spectral_norm(lin)
            layers += [lin, nn.Dropout(0.2), get_activation(act)]
            last = h
        out = nn.Linear(last, 1)
        if spectral_norm:
            out = nn.utils.spectral_norm(out)
        layers += [out, nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ------------------------------
# ALAD
# ------------------------------
class ALAD(BaseVisionDeepDetector):
    """ALAD，继承 BaseVisionDeepDetector。
    关键接口：
      - build_model(): 构建 Enc/Dec 与判别器，以及优化器
      - training_forward(batch): 单个 batch 的对抗训练
      - evaluating_forward(batch): 返回该 batch 的异常分数 (numpy array)
    """

    def __init__(self,
                 # ALAD 通用参数
                 activation_hidden_gen: str = 'tanh',
                 activation_hidden_disc: str = 'tanh',
                 output_activation: Optional[str] = None,
                 dropout_rate: float = 0.2,
                 latent_dim: int = 128,
                 disc_xx_layers: List[int] = [256, 128, 64],
                 disc_xz_layers: List[int] = [256, 128, 64],
                 disc_zz_layers: List[int] = [128, 64],
                 learning_rate_gen: float = 1e-4,
                 learning_rate_disc: float = 1e-4,
                 add_recon_loss: bool = False,
                 lambda_recon_loss: float = 0.1,
                 add_disc_zz_loss: bool = True,
                 spectral_normalization: bool = False,
                 # CNN 相关
                 enc_base_ch: int = 64,
                 dec_base_ch: int = 64,
                 xx_base_ch: int = 64,
                 xz_feat_dim: int = 256,
                 contamination: float = 0.1,
                 preprocessing: bool = True,
                 lr: float = 1e-3,
                 epoch_num: int = 10,
                 batch_size: int = 16,
                 optimizer_name: str = 'adam',
                 criterion_name: str = 'mse',
                 device: Optional[str] = None,
                 random_state: int = 42,
                 verbose: int = 1,
                 train_transform=None,
                 eval_transform=None,
                 **kwargs):
        super().__init__(contamination=contamination,
                         preprocessing=preprocessing,
                         lr=lr,
                         epoch_num=epoch_num,
                         batch_size=batch_size,
                         optimizer_name=optimizer_name,
                         criterion_name=criterion_name,
                         device=device,
                         random_state=random_state,
                         verbose=verbose,
                         train_transform=train_transform,
                         eval_transform=eval_transform,
                         **kwargs)
        # 保存超参
        self.activation_hidden_gen = activation_hidden_gen
        self.activation_hidden_disc = activation_hidden_disc
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.disc_xx_layers = disc_xx_layers
        self.disc_xz_layers = disc_xz_layers
        self.disc_zz_layers = disc_zz_layers
        self.learning_rate_gen = learning_rate_gen
        self.learning_rate_disc = learning_rate_disc
        self.add_recon_loss = add_recon_loss
        self.lambda_recon_loss = lambda_recon_loss
        self.add_disc_zz_loss = add_disc_zz_loss
        self.spectral_normalization = spectral_normalization

        self.enc_base_ch = enc_base_ch
        self.dec_base_ch = dec_base_ch
        self.xx_base_ch = xx_base_ch
        self.xz_feat_dim = xz_feat_dim
        self.enc: Optional[nn.Module] = None
        self.dec: Optional[nn.Module] = None
        self.disc_xx: Optional[nn.Module] = None
        self.img_feat: Optional[nn.Module] = None
        self.disc_xz: Optional[nn.Module] = None
        self.disc_zz: Optional[nn.Module] = None

        # 优化器
        self.opt_gen: Optional[optim.Optimizer] = None
        self.opt_disc: Optional[optim.Optimizer] = None

        # 历史损失
        self.hist_loss_disc: List[float] = []
        self.hist_loss_gen: List[float] = []


    def build_model(self):
        device = self.device
        # 生成器
        self.enc = ConvEncoder(self.latent_dim, in_ch=3, base_ch=self.enc_base_ch).to(device)
        self.dec = ConvDecoder(self.latent_dim, out_ch=3, base_ch=self.dec_base_ch).to(device)

        # 判别器们
        self.disc_xx = DiscXX(in_ch_pair=6, base_ch=self.xx_base_ch).to(device)
        self.img_feat = ImgFeat(in_ch=3, base_ch=64, feat_dim=self.xz_feat_dim).to(device)
        self.disc_xz = MLPDisc(in_dim=self.xz_feat_dim + self.latent_dim,
                               hidden=self.disc_xz_layers,
                               act=self.activation_hidden_disc,
                               spectral_norm=self.spectral_normalization).to(device)
        self.disc_zz = MLPDisc(in_dim=2 * self.latent_dim,
                               hidden=self.disc_zz_layers,
                               act=self.activation_hidden_disc,
                               spectral_norm=self.spectral_normalization).to(device)

        # 优化器：判别器和生成器分开
        disc_params = list(self.disc_xx.parameters()) + \
                      list(self.disc_xz.parameters()) + \
                      list(self.disc_zz.parameters())
        gen_params = list(self.enc.parameters()) + list(self.dec.parameters())

        self.opt_disc = optim.Adam(disc_params, lr=self.learning_rate_disc, betas=(0.5, 0.999))
        self.opt_gen = optim.Adam(gen_params, lr=self.learning_rate_gen, betas=(0.5, 0.999))

        return nn.Module()

    def training_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """执行一个 batch 的对抗训练（判别器一步 + 生成器一步）。
        返回总损失（float）用于日志；优化在本函数内完成。
        """
        self.enc.train()
        self.dec.train()
        self.disc_xx.train()
        self.disc_xz.train()
        self.disc_zz.train()
        x_real, _ = batch
        x_real = x_real.to(self.device)
        B = x_real.size(0)

        # 采样潜变量 z ~ N(0, I)
        z_real = torch.randn(B, self.latent_dim, device=self.device)

        bce = nn.BCELoss()

        #  更新判别器
        self.opt_disc.zero_grad(set_to_none=True)

        with torch.no_grad():
            x_gen_d = self.dec(z_real)
            z_gen_d = self.enc(x_real)

        # D_xz: (x, z_gen) 为真；(x_gen, z_real) 为假
        feat_x = self.img_feat(x_real)
        feat_x_gen = self.img_feat(x_gen_d)
        out_true_xz = self.disc_xz(torch.cat([feat_x, z_gen_d], dim=1))
        out_fake_xz = self.disc_xz(torch.cat([feat_x_gen, z_real], dim=1))

        # D_xx: (x, x) 为真；(x, x_gen) 为假
        out_true_xx = self.disc_xx(x_real, x_real)
        out_fake_xx = self.disc_xx(x_real, x_gen_d)

        ones = torch.ones_like(out_true_xz)
        zeros = torch.zeros_like(out_true_xz)

        loss_dxz = bce(out_true_xz, ones) + bce(out_fake_xz, zeros)
        loss_dxx = bce(out_true_xx, torch.ones_like(out_true_xx)) + \
                   bce(out_fake_xx, torch.zeros_like(out_fake_xx))

        if self.add_disc_zz_loss:
            out_true_zz = self.disc_zz(torch.cat([z_real, z_real], dim=1))
            out_fake_zz = self.disc_zz(torch.cat([z_real, z_gen_d], dim=1))
            loss_dzz = bce(out_true_zz, torch.ones_like(out_true_zz)) + \
                       bce(out_fake_zz, torch.zeros_like(out_fake_zz))
            loss_disc = loss_dxz + loss_dxx + loss_dzz
        else:
            loss_disc = loss_dxz + loss_dxx

        loss_disc.backward()
        self.opt_disc.step()

        # 更新生成器（Enc+Dec）
        self.opt_gen.zero_grad(set_to_none=True)

        x_gen = self.dec(z_real)
        z_gen = self.enc(x_real)

        feat_x = self.img_feat(x_real)
        feat_x_gen = self.img_feat(x_gen)

        out_true_xz = self.disc_xz(torch.cat([feat_x, z_gen], dim=1))
        out_fake_xz = self.disc_xz(torch.cat([feat_x_gen, z_real], dim=1))
        out_true_xx = self.disc_xx(x_real, x_real)
        out_fake_xx = self.disc_xx(x_real, x_gen)

        # 生成器希望“骗过”判别器：伪造判为真、真实判为假
        loss_gexz = bce(out_fake_xz, ones) + bce(out_true_xz, zeros)
        loss_gexx = bce(out_fake_xx, torch.ones_like(out_fake_xx)) + \
                    bce(out_true_xx, torch.zeros_like(out_true_xx))

        if self.add_disc_zz_loss:
            out_true_zz = self.disc_zz(torch.cat([z_real, z_real], dim=1))
            out_fake_zz = self.disc_zz(torch.cat([z_real, z_gen], dim=1))
            loss_gezz = bce(out_fake_zz, torch.ones_like(out_fake_zz)) + \
                        bce(out_true_zz, torch.zeros_like(out_true_zz))
            cycle_consistency = loss_gezz + loss_gexx
            loss_gen = loss_gexz + cycle_consistency
        else:
            cycle_consistency = loss_gexx
            loss_gen = loss_gexz + cycle_consistency

        if self.add_recon_loss:
            x_recon = self.dec(self.enc(x_real))
            loss_recon = F.mse_loss(x_recon, x_real)
            loss_gen = loss_gen + self.lambda_recon_loss * loss_recon

        loss_gen.backward()
        self.opt_gen.step()

        # 记录历史
        self.hist_loss_disc.append(float(loss_disc.item()))
        self.hist_loss_gen.append(float(loss_gen.item()))

        # 返回一个合并损失，供父类日志显示
        return float((loss_disc + loss_gen).item())


    @torch.no_grad()
    def evaluating_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        self.enc.eval();
        self.dec.eval()
        self.disc_xx.eval();
        self.disc_xz.eval();
        self.disc_zz.eval()

        x, _ = batch
        x = x.to(self.device)
        z = self.enc(x)
        x_hat = self.dec(z)

        # 使用 D_xx 的“真/伪”差异作为分数
        out_true_xx = self.disc_xx(x, x)
        out_fake_xx = self.disc_xx(x, x_hat)
        score = torch.mean(torch.abs((out_true_xx - out_fake_xx) ** 2), dim=1)  # [B]
        return score.detach().cpu().numpy()

    # 可选：绘制学习曲线（与原 ALAD 一致）
    def get_history(self) -> Dict[str, List[float]]:
        return {"loss_disc": self.hist_loss_disc, "loss_gen": self.hist_loss_gen}



if __name__ == '__main__':
    # 准备训练/评估的图像路径列表 train_paths / test_paths
    train_paths: List[str] = [
        # '/path/to/img1.jpg', '/path/to/img2.jpg', ...
    ]
    test_paths: List[str] = [
        # '/path/to/test1.jpg', '/path/to/test2.jpg', ...
    ]
    # 2) 初始化并训练
    model = ALAD(epoch_num=5, batch_size=8, verbose=1,
                       preprocessing=True, latent_dim=128)
    model.fit(train_paths)
    # 3) 计算分数
    scores = model.decision_function(test_paths)
    print("scores shape:", np.array(scores).shape)

    # 4) 阈值与标签（由 BaseVisionDeepDetector/_process_decision_scores 提供）
    #    model.labels_ / model.threshold_ 可用
