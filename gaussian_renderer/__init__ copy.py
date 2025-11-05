#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import numpy as np
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel



# 根据提供的 decode_segmap 定义颜色映射
def get_class_colors(n_classes=6):
    """定义颜色映射 - 根据类别数量自定义"""
    label_colours = np.array([
        [0, 0, 0], [0,0,255], [0, 255, 0], [255, 0, 0],
        [0, 255, 255], [255,255,0]
    ])
    
    
    # 转换为浮点数并归一化到 [0,1]
    label_colours = label_colours.astype(np.float32) / 255.0
    
    return torch.tensor(label_colours, dtype=torch.float32, device='cuda')

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])


    # 获取语义特征 - 每个锚点一个特征向量
    semantic_input = torch.cat([feat, ob_view], dim=1)
    if pc.appearance_dim > 0:
        semantic_input = torch.cat([semantic_input, appearance], dim=1)
    semantic = pc.get_semantic_mlp(semantic_input)  # [N, C]
    
    # 将语义特征复制给每个偏移点
    semantic = semantic.unsqueeze(1).repeat(1, pc.n_offsets, 1)  # [N, k, C]
    semantic = semantic.reshape(anchor.shape[0] * pc.n_offsets, pc.num_semantic_classes)  # [N*k, C]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, semantic, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, semantic, scale_rot, offsets = masked.split(
        [6, 3, 3, pc.num_semantic_classes, 7, 3], dim=-1
    )
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, semantic
    else:
        return xyz, color, opacity, scaling, rot, semantic

def semantic_prob_to_color(semantic_prob, class_colors):
    """
    将语义概率图转换为彩色图像
    :param semantic_prob: [C, H, W] 语义概率图
    :param class_colors: [n_classes, 3] 类别颜色映射表
    :return: [3, H, W] RGB彩色图像
    """
    # 获取最可能的类别索引
    class_indices = torch.argmax(semantic_prob, dim=0)  # [H, W]
    
    # 将类别索引映射为颜色
    height, width = class_indices.shape
    color_image = torch.zeros(3, height, width, device=semantic_prob.device, dtype=torch.float32)
    
    for class_idx in range(class_colors.shape[0]):
        mask = (class_indices == class_idx)
        color_image[:, mask] = class_colors[class_idx][:, None]
    
    return color_image

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # 获取类别颜色映射
    class_colors = get_class_colors(pc.num_semantic_classes)
    
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, semantic = generate_neural_gaussians(
            viewpoint_camera, pc, visible_mask, is_training=is_training
        )
    else:
        xyz, color, opacity, scaling, rot, semantic = generate_neural_gaussians(
            viewpoint_camera, pc, visible_mask, is_training=is_training
        )
    

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration for color
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # 渲染语义分割图 - 每个通道单独渲染
    semantic_outputs = []
    for i in range(pc.num_semantic_classes):
        # 提取当前通道的语义特征
        semantic_channel = semantic[:, i].unsqueeze(1)  # [N, 1]
        # 复制为3通道（因为光栅化器需要3通道）
        semantic_color = semantic_channel.expand(-1, 3)  # [N, 3]

        # 设置光栅化器
        semantic_raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros(3, device=bg_color.device),  # 背景设为0
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        
        semantic_rasterizer = GaussianRasterizer(raster_settings=semantic_raster_settings)
        
        # 渲染当前通道
        rendered_semantic_channel, _ = semantic_rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = semantic_color,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)
        
        # 只取第一个通道（所有通道值相同）
        semantic_outputs.append(rendered_semantic_channel[0:1])
    
    # 组合语义通道
    rendered_semantic = torch.cat(semantic_outputs, dim=0)  # [C, H, W]
    
    # 获取预测类别索引
    class_indices = torch.argmax(rendered_semantic, dim=0)  # [H, W]

    # print(f"class_colors shape: {class_colors.shape}")  # 应该是 [num_classes, 3]
    # print(f"class_indices min: {class_indices.min().item()}, max: {class_indices.max().item()}")

    # 将类别索引映射为彩色图像
    semantic_color = class_colors[class_indices].permute(2, 0, 1)  # [3, H, W]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {
            "render": rendered_image,
            "semantic_logits": rendered_semantic,  # 保留概率图用于训练
            "semantic_color": semantic_color,  # 彩色语义图
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
        }
    else:
        return {
            "render": rendered_image,
            "semantic_logits": rendered_semantic,  # 返回彩色语义图
            "semantic_color": semantic_color,  # 返回彩色语义图
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
        }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(
        means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    return radii_pure > 0