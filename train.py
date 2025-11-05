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

import os
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F
import psutil
import csv
import datetime

# 添加内存和显存监控类
class SystemMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.max_memory_usage = 0
        self.max_gpu_memory_usage = 0
        
    def update_memory_usage(self):
        # 系统内存使用量（MB）
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        self.max_memory_usage = max(self.max_memory_usage, memory_mb)
        
        # GPU内存使用量（MB）
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.max_gpu_memory_usage = max(self.max_gpu_memory_usage, gpu_memory)
    
    def get_stats(self):
        return {
            'max_memory_mb': self.max_memory_usage,
            'max_gpu_memory_mb': self.max_gpu_memory_usage
        }
    
    def reset(self):
        self.max_memory_usage = 0
        self.max_gpu_memory_usage = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

# 添加CSV记录函数
def save_metrics_to_csv(metrics_dict, csv_file='Scaffold-GS_metrics.csv'):
    """保存指标到CSV文件"""
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

def get_class_colors(n_classes=9):
    label_colours = np.array([
        [0, 0, 0], [0,0,255], [0, 255, 0], [255, 0, 0],
        [0, 255, 255], [255,255,0],[255,0,255],[255,255,255],[0,125,0]
    ], dtype=np.float32)  # 确保使用float32
    
    return label_colours[:n_classes] / 255.0

def rgb_to_class_index(rgb_image, class_colors):
    """
    将RGB语义图像转换为类别索引图
    :param rgb_image: [3, H, W] 语义RGB图像
    :param class_colors: [n_classes, 3] 类别颜色表
    :return: [H, W] 类别索引图
    """
    # 确保输入是float32类型
    if rgb_image.dtype == torch.float64:
        rgb_image = rgb_image.float()  # 转换为float32
    
    # 将图像和颜色表转为相同形状
    rgb_image = rgb_image.permute(1, 2, 0)  # [H, W, 3]
    
    # 确保class_colors是torch tensor且在GPU上
    if not isinstance(class_colors, torch.Tensor):
        class_colors = torch.tensor(class_colors, device=rgb_image.device, dtype=torch.float32)
    elif class_colors.dtype == torch.float64:
        class_colors = class_colors.float()  # 转换为float32
    
    # 计算每个像素与所有颜色的距离
    distances = torch.cdist(
        rgb_image.reshape(-1, 3),  # [H*W, 3]
        class_colors                # [n_classes, 3]
    )  # 结果 [H*W, n_classes]
    
    # 找到最小距离对应的类别索引
    class_indices = torch.argmin(distances, dim=1)  # [H*W]
    
    # 恢复原始形状
    return class_indices.view(rgb_image.shape[0], rgb_image.shape[1])  # [H, W]

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None, monitor=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, 
                              dataset.num_semantic_classes)  # 添加语义类别数
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity, semantic = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["semantic_logits"]

        gt_image = viewpoint_cam.original_image.cuda()
        gt_semantic = viewpoint_cam.semantic_mask.cuda()  # 获取语义真值
        
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        
        # 语义损失
        semantic_weight = 0.01 * min(1.0, max(0.0, (iteration - 5000) / 15000))
        # 将RGB真值转换为类别索引图
        if gt_semantic.dtype == torch.float64:
            gt_semantic = gt_semantic.float()

        # 转换真值为类别索引
        class_colors = get_class_colors(gaussians.num_semantic_classes)
        gt_semantic_index = rgb_to_class_index(gt_semantic, class_colors)

        # 计算语义损失
        semantic_loss = torch.nn.functional.cross_entropy(
            semantic.unsqueeze(0),  # [1, C, H, W]
            gt_semantic_index.unsqueeze(0),  # [1, H, W]
            label_smoothing=0.1,
            reduction="mean"
        )

        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg + semantic_weight*semantic_loss  # 添加语义损失
        
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger, semantic_loss)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            # 更新内存使用情况
            if monitor and iteration % 100 == 0:  # 每100次迭代更新一次
                monitor.update_memory_usage()

    # 记录训练结束时间和内存使用
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    if monitor:
        stats = monitor.get_stats()
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Max memory usage: {stats['max_memory_mb']:.2f} MB")
        logger.info(f"Max GPU memory usage: {stats['max_gpu_memory_mb']:.2f} MB")
    
    return training_time

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, semantic_loss=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)
        if semantic_loss is not None:
            tb_writer.add_scalar(f'{dataset_name}/semantic_loss', semantic_loss.item(), iteration)

    if wandb is not None:
        log_data = {"train_l1_loss": Ll1, 'train_total_loss': loss}
        if semantic_loss is not None:
            log_data["semantic_loss"] = semantic_loss
        wandb.log(log_data)
    
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                semantic_acc = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []
                    semantic_pred_list = []
                    semantic_gt_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    semantic_logits = render_pkg["semantic_logits"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_semantic = viewpoint.semantic_mask.to("cuda")
                    
                    # 计算语义准确率
                    pred_labels = semantic_logits.argmax(dim=0)  # [H, W]
                    gt_semantic = viewpoint.semantic_mask.cuda()  # 获取语义真值
                    
                    # 转换真值为类别索引
                    class_colors = get_class_colors(scene.gaussians.num_semantic_classes)
                    gt_semantic_index = rgb_to_class_index(gt_semantic, class_colors)  # [H, W]
                    
                    # 计算准确率
                    semantic_acc += (pred_labels == gt_semantic_index).float().mean().item()
                    
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)
                        
                        # 修复语义分割图像记录 - 添加通道维度
                        semantic_display = pred_labels[None].float() / (scene.gaussians.num_semantic_classes - 1)  # [1, H, W]
                        semantic_display = semantic_display.unsqueeze(1)  # 添加通道维度 [1, 1, H, W]
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/semantic".format(viewpoint.image_name), semantic_display, global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            # 同样修复wandb的语义图像
                            semantic_wandb = pred_labels.float() / (scene.gaussians.num_semantic_classes - 1)
                            semantic_pred_list.append(semantic_wandb.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            
                            # 修复语义真值图像记录
                            semantic_gt_display = gt_semantic_index[None].float() / (scene.gaussians.num_semantic_classes - 1)  # [1, H, W]
                            semantic_gt_display = semantic_gt_display.unsqueeze(1)  # 添加通道维度 [1, 1, H, W]
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/semantic_gt".format(viewpoint.image_name), semantic_gt_display, global_step=iteration)
                            
                            if wandb:
                                gt_image_list.append(gt_image[None])
                                semantic_gt_wandb = gt_semantic_index.float() / (scene.gaussians.num_semantic_classes - 1)
                                semantic_gt_list.append(semantic_gt_wandb.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                semantic_acc /= len(config['cameras'])
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} Semantic Acc {}".format(iteration, config['name'], l1_test, psnr_test, semantic_acc))

                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/semantic_accuracy', semantic_acc, iteration)
                if wandb is not None:
                    wandb.log({
                        f"{config['name']}_loss_viewpoint_l1_loss": l1_test,
                        f"{config['name']}_PSNR": psnr_test,
                        f"{config['name']}_semantic_accuracy": semantic_acc
                    })

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    semantic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantic")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(semantic_path, exist_ok=True)
    
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        semantic = render_pkg["semantic_color"]
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        gt = view.original_image[0:3, :, :]
        
        errormap = (rendering - gt).abs()

        # 获取原始图像名称
        if hasattr(view, 'image_name') and view.image_name:
            # 如果有image_name属性，使用原始图像名称
            base_name = os.path.splitext(view.image_name)[0]  # 去掉扩展名
            name_str = base_name + ".png"
        elif hasattr(view, 'image_path') and view.image_path:
            # 如果有image_path属性，从路径中提取文件名
            base_name = os.path.splitext(os.path.basename(view.image_path))[0]
            name_str = base_name + ".png"
        else:
            # 如果都没有，回退到数字索引
            name_str = '{0:05d}'.format(idx) + ".png"
        
        name_list.append(name_str)
        torchvision.utils.save_image(rendering, os.path.join(render_path, name_str))
        torchvision.utils.save_image(errormap, os.path.join(error_path, name_str))
        torchvision.utils.save_image(gt, os.path.join(gts_path, name_str))
        torchvision.utils.save_image(semantic, os.path.join(semantic_path, name_str))
        per_view_dict[name_str] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=False, skip_test=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
                              dataset.num_semantic_classes)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        visible_count = None
        
        if not skip_train:
            visible_count = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
    
    return visible_count

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def readSemanticImages(semantic_dir, segment_dir, image_names, num_classes):
    """读取语义预测和真值图像"""
    semantic_preds = []
    semantic_gts = []
    
    for fname in image_names:
        # 读取预测的语义图像
        semantic_path = semantic_dir / fname
        if os.path.exists(semantic_path):
            semantic_pred = Image.open(semantic_path)
            semantic_pred = tf.to_tensor(semantic_pred).unsqueeze(0).cuda()
            semantic_preds.append(semantic_pred)
        
        # 读取真值语义图像（从segment文件夹）
        base_name = os.path.splitext(fname)[0]
        gname = base_name + "_color.png"
        segment_path = segment_dir / gname
        if not os.path.exists(segment_path):
            # 如果segment文件夹中不存在，尝试在gt的semantic子文件夹中查找
            segment_path = segment_dir.parent / "semantic_gt" / gname
        
        if os.path.exists(segment_path):
            semantic_gt = Image.open(segment_path)
            semantic_gt = tf.to_tensor(semantic_gt).unsqueeze(0).cuda()
            semantic_gts.append(semantic_gt)
        else:
            # 如果找不到真值，创建一个空的tensor
            print("No GT!")
            semantic_gts.append(None)
    
    return semantic_preds, semantic_gts

def compute_semantic_metrics(preds, gts, num_classes):
    """计算语义分割指标：准确率和mIOU"""
    total_accuracy = 0.0
    total_iou = 0.0
    count = 0
    
    class_colors = get_class_colors(num_classes)
    
    for pred, gt in zip(preds, gts):
        if gt is None:
            continue
            
        # 将预测和真值转换为类别索引
        pred_labels = rgb_to_class_index(pred.squeeze(0), class_colors)
        gt_labels = rgb_to_class_index(gt.squeeze(0), class_colors)
        
        # 计算准确率
        accuracy = (pred_labels == gt_labels).float().mean().item()
        total_accuracy += accuracy
        
        # 计算每个类别的IOU
        ious = []
        for class_id in range(num_classes):
            pred_mask = (pred_labels == class_id)
            gt_mask = (gt_labels == class_id)
            
            intersection = (pred_mask & gt_mask).sum().float()
            union = (pred_mask | gt_mask).sum().float()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou.item())
            else:
                ious.append(0.0)
        
        # 计算mIOU（忽略背景类0）
        if len(ious) > 1:
            miou = sum(ious[1:]) / (len(ious) - 1)  # 忽略背景类
        else:
            miou = 0.0
            
        total_iou += miou
        count += 1
    
    if count == 0:
        return 0.0, 0.0
    
    mean_accuracy = total_accuracy / count
    mean_iou = total_iou / count
    
    return mean_accuracy, mean_iou

def evaluate(model_paths,source_path, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None, evaluation_time=None, num_semantic_classes=9):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"
    exp_name = model_paths.split('/')[-2]
    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        semantic_dir = method_dir / "semantic"
        
        # 查找segment文件夹（语义真值）
        segment_dir = Path(source_path).absolute()/"segment"  # segment文件夹在数据集的根目录
        print(segment_dir)
        renders, gts, image_names = readImages(renders_dir, gt_dir)
        
        # 读取语义图像
        semantic_preds, semantic_gts = readSemanticImages(semantic_dir, segment_dir, image_names, num_semantic_classes)
        
        ssims = []
        psnrs = []
        lpipss = []
        semantic_accs = []
        semantic_mious = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
            
            # 计算语义指标
            if idx < len(semantic_preds) and semantic_gts[idx] is not None:
                accuracy, miou = compute_semantic_metrics([semantic_preds[idx]], [semantic_gts[idx]], num_semantic_classes)
                semantic_accs.append(accuracy)
                semantic_mious.append(miou)

        # 计算平均语义指标
        mean_semantic_acc = torch.tensor(semantic_accs).mean().item() if semantic_accs else 0.0
        mean_semantic_miou = torch.tensor(semantic_mious).mean().item() if semantic_mious else 0.0

        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })
            if semantic_accs:
                wandb.log({"test_semantic_accuracy": mean_semantic_acc})
                wandb.log({"test_semantic_miou": mean_semantic_miou})

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        logger.info("  Semantic Acc: \033[1;35m{:>12.7f}\033[0m".format(mean_semantic_acc, ".5"))
        logger.info("  Semantic mIOU: \033[1;35m{:>12.7f}\033[0m".format(mean_semantic_miou, ".5"))
        print("")

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/Semantic_Accuracy', mean_semantic_acc, 0)
            tb_writer.add_scalar(f'{dataset_name}/Semantic_mIOU', mean_semantic_miou, 0)
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
            "Semantic_Accuracy": mean_semantic_acc,
            "Semantic_mIOU": mean_semantic_miou
        })
        per_view_dict[scene_dir][method].update({
            "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            "Semantic_Accuracy": {name: acc for acc, name in zip(semantic_accs, image_names)},
            "Semantic_mIOU": {name: miou for miou, name in zip(semantic_mious, image_names)},
            "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}
        })

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
    # 返回评估指标用于保存到CSV
    return {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
        "Semantic_Accuracy": mean_semantic_acc,
        "Semantic_mIOU": mean_semantic_miou,
        "Evaluation_Time": evaluation_time or 0.0
    }

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 初始化系统监控器
    monitor = SystemMonitor()
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 训练阶段
    training_time = training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger, monitor=monitor)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training_time += training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path, monitor=monitor)

    logger.info("\nTraining complete.")

    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args))
    logger.info("\nRendering complete.")

    logger.info("\n Starting evaluation...")
    evaluation_start_time = time.time()
    # 使用实际的语义类别数进行评估
    eval_metrics = evaluate(args.model_path, args.source_path,visible_count=visible_count, wandb=wandb, logger=logger, num_semantic_classes=lp.extract(args).num_semantic_classes)
    evaluation_time = time.time() - evaluation_start_time
    eval_metrics["Evaluation_Time"] = evaluation_time
    logger.info("\nEvaluating complete.")
    
    # 计算总时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # 获取内存使用统计
    stats = monitor.get_stats()
    
    # 准备要保存的指标
    metrics_dict = {
        'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Dataset': dataset,
        'Experiment': exp_name,
        'Total_Time_Seconds': total_time,
        'Training_Time_Seconds': training_time,
        'Evaluation_Time_Seconds': evaluation_time,
        'Max_Memory_MB': stats['max_memory_mb'],
        'Max_GPU_Memory_MB': stats['max_gpu_memory_mb'],
        'SSIM': eval_metrics.get('SSIM', 0),
        'PSNR': eval_metrics.get('PSNR', 0),
        'LPIPS': eval_metrics.get('LPIPS', 0),
        'Semantic_Accuracy': eval_metrics.get('Semantic_Accuracy', 0),
        'Semantic_mIOU': eval_metrics.get('Semantic_mIOU', 0)
    }
    
    # 保存指标到CSV
    save_metrics_to_csv(metrics_dict, 'Scaffold-GS_metrics.csv')
    
    logger.info(f"\n=== Performance Summary ===")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Evaluation time: {evaluation_time:.2f} seconds")
    logger.info(f"Max memory usage: {stats['max_memory_mb']:.2f} MB")
    logger.info(f"Max GPU memory usage: {stats['max_gpu_memory_mb']:.2f} MB")
    logger.info(f"Semantic Accuracy: {eval_metrics.get('Semantic_Accuracy', 0):.4f}")
    logger.info(f"Semantic mIOU: {eval_metrics.get('Semantic_mIOU', 0):.4f}")
    logger.info(f"Metrics saved to Scaffold-GS_metrics.csv")