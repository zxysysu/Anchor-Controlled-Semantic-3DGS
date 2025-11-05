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
import torch.nn.functional as F  # 添加这行
from torchvision.transforms.functional import resize
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import os
from PIL import Image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1080:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # 处理原始图像
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    # 处理语义信息 - 更安全的处理方式
    semantic_mask = None
    if hasattr(cam_info, 'semantic_image_path') and cam_info.semantic_image_path:
        print(f"Loading semantic image: {cam_info.semantic_image_path}")
        if os.path.exists(cam_info.semantic_image_path):
            try:
                # 加载语义图片
                semantic_img = Image.open(cam_info.semantic_image_path)
                # 调整语义图片大小，使用最近邻插值保持颜色值
                semantic_img = semantic_img.resize(resolution, Image.NEAREST)
                
                # 转换为张量并归一化到 [0, 1]
                semantic_mask = PILtoTorch(semantic_img, resolution)
                
                # 确保是三通道
                if semantic_mask.shape[0] == 1:  # 如果是单通道
                    semantic_mask = semantic_mask.repeat(3, 1, 1)  # 复制为三通道
                elif semantic_mask.shape[0] == 4:  # 如果是RGBA
                    semantic_mask = semantic_mask[:3, ...]  # 取前三个通道
            except Exception as e:
                print(f"Error loading semantic image {cam_info.semantic_image_path}: {str(e)}")
                semantic_mask = None
        else:
            print(f"Semantic image not found: {cam_info.semantic_image_path}")
            semantic_mask = None
    else:
        print(f"No semantic image path provided for camera {cam_info.image_name}")
        semantic_mask = None

    return Camera(
        colmap_id=cam_info.uid, 
        R=cam_info.R, 
        T=cam_info.T, 
        FoVx=cam_info.FovX, 
        FoVy=cam_info.FovY, 
        image=gt_image, 
        gt_alpha_mask=loaded_mask,
        semantic_mask=semantic_mask,  # 可能为 None
        image_name=cam_info.image_name, 
        uid=id, 
        data_device=args.data_device
    )
# # 修改 loadCam 函数
# def loadCam(args, id, cam_info, resolution_scale):
#     orig_w, orig_h = cam_info.image.size

#     if args.resolution in [1, 2, 4, 8]:
#         resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
#     else:  # should be a type that converts to float
#         if args.resolution == -1:
#             if orig_w > 1600:
#                 global WARNED
#                 if not WARNED:
#                     print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
#                         "If this is not desired, please explicitly specify '--resolution/-r' as 1")
#                     WARNED = True
#                 global_down = orig_w / 1600
#             else:
#                 global_down = 1
#         else:
#             global_down = orig_w / args.resolution

#         scale = float(global_down) * float(resolution_scale)
#         resolution = (int(orig_w / scale), int(orig_h / scale))

#     # 处理原始图像
#     resized_image_rgb = PILtoTorch(cam_info.image, resolution)
#     gt_image = resized_image_rgb[:3, ...]
#     loaded_mask = None

#     if resized_image_rgb.shape[1] == 4:
#         loaded_mask = resized_image_rgb[3:4, ...]
    
#      # 处理语义信息
#     semantic_mask = None
#     print(cam_info.semantic_image_path)
#     # 检查语义图片路径是否存在
#     if hasattr(cam_info, 'semantic_image_path') and cam_info.semantic_image_path and os.path.exists(cam_info.semantic_image_path):
#         try:
#             # 加载语义图片
#             semantic_img = Image.open(cam_info.semantic_image_path)
#             # 调整语义图片大小，使用最近邻插值保持颜色值
#             semantic_img = semantic_img.resize(resolution, Image.NEAREST)
            
#             # 转换为张量并归一化到 [0, 1]
#             semantic_mask = PILtoTorch(semantic_img, resolution)
            
#             # 确保是三通道
#             if semantic_mask.shape[0] == 1:  # 如果是单通道
#                 semantic_mask = semantic_mask.repeat(3, 1, 1)  # 复制为三通道
#             elif semantic_mask.shape[0] == 4:  # 如果是RGBA
#                 semantic_mask = semantic_mask[:3, ...]  # 取前三个通道
#         except Exception as e:
#             print(f"Error loading semantic image {cam_info.semantic_image_path}: {str(e)}")
#             semantic_mask = None
#     else:
#         print(f"Semantic image not found: {cam_info.semantic_image_path}")
#         semantic_mask = None

#     return Camera(
#         colmap_id=cam_info.uid, 
#         R=cam_info.R, 
#         T=cam_info.T, 
#         FoVx=cam_info.FovX, 
#         FoVy=cam_info.FovY, 
#         image=gt_image, 
#         gt_alpha_mask=loaded_mask,
#         semantic_mask=semantic_mask,  # 这里可能是 None
#         image_name=cam_info.image_name, 
#         uid=id, 
#         data_device=args.data_device
#     )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


