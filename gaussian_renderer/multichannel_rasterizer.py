import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class MultiChannelGaussianRasterizer:
    def __init__(self, raster_settings):
        self.raster_settings = raster_settings
        # 为每个通道创建单独的光栅化器
        self.rasterizers = [
            GaussianRasterizer(raster_settings) 
            for _ in range(raster_settings.output_channels)
        ]
    
    def __call__(self, means3D, means2D, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp):
        # 确保颜色预处理有正确的维度
        assert colors_precomp.shape[1] == self.raster_settings.output_channels
        
        # 为每个通道创建输出
        outputs = []
        radii_list = []
        
        for i in range(self.raster_settings.output_channels):
            # 为当前通道提取颜色
            channel_colors = colors_precomp[:, i].unsqueeze(1).repeat(1, 3)
            
            # 渲染当前通道
            rendered, radii = self.rasterizers[i](
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=channel_colors,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp
            )
            
            # 只取第一个通道（所有通道值相同）
            outputs.append(rendered[0:1])
            radii_list.append(radii)
        
        # 合并所有通道
        rendered_semantic = torch.cat(outputs, dim=0)  # [C, H, W]
        # 使用第一个通道的radii（所有通道相同）
        radii = radii_list[0]
        
        return rendered_semantic, radii