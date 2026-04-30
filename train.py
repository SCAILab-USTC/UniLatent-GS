import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


import torch.nn.functional as F
from types import SimpleNamespace
import torchvision.transforms as transforms
import numpy as np
import trimesh


from miche_integration import load_miche_model, MICHEEncoder

def get_standard_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_miche_encoder(args):
    print(f"Loading MICHE Encoder...")
    
    try:
        # Enable CLIP image encoding if multimodal loss is enabled
        enable_clip = args.lambda_multimodal > 0
        encoder = MICHEEncoder(
            ckpt_path=args.miche_ckpt_path,
            enable_clip_image=enable_clip,
            clip_model_version=args.clip_model_version
        )
        model = encoder.load_model()
                
        return model
    except Exception as e:
        print(f"Failed to load MICHE model: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_miche_pointcloud(gaussians, npoints=4096):
    xyz = gaussians.get_xyz
    total_points = xyz.shape[0]
    
    if total_points >= npoints:
        idx = torch.randperm(total_points, device="cuda")[:npoints]
    else:
        idx = torch.arange(total_points, device="cuda")
        
    sampled_xyz = xyz[idx]
    
    shs_dc = gaussians._features_dc[idx]
    rgb = 0.5 + 0.28209479177387814 * shs_dc.squeeze(1)
    sampled_rgb = torch.clamp(rgb, min=0.0, max=1.0)
    
    centroid = torch.mean(sampled_xyz, dim=0, keepdim=True)
    sampled_xyz = sampled_xyz - centroid
    dist = torch.max(torch.sqrt(torch.sum(sampled_xyz ** 2, dim=1)), dim=0)[0]
    sampled_xyz = sampled_xyz / (dist + 1e-5)
    
    pc = torch.cat([sampled_xyz, sampled_rgb], dim=1)
    pc = pc.unsqueeze(0) 
    
    return pc

def get_masked_backprojected_pc(image, depth, camera, downsample_factor=8, mask_threshold=0.5):
    if camera.gt_alpha_mask is not None:
        mask = camera.gt_alpha_mask.to(image.device)
    else:
        mask = torch.ones_like(depth)

    img_tensor = image.unsqueeze(0)
    depth_tensor = depth.unsqueeze(0)
    mask_tensor = mask.unsqueeze(0)
    
    img_small = F.avg_pool2d(img_tensor, kernel_size=downsample_factor, stride=downsample_factor)
    depth_small = F.avg_pool2d(depth_tensor, kernel_size=downsample_factor, stride=downsample_factor)
    mask_small = F.avg_pool2d(mask_tensor, kernel_size=downsample_factor, stride=downsample_factor)
    
    rgb_s = img_small.squeeze(0)
    depth_s = depth_small.squeeze(0)
    mask_s = mask_small.squeeze(0)
    
    H_s, W_s = depth_s.shape[1], depth_s.shape[2]
    
    valid_mask = (mask_s > mask_threshold).reshape(-1)
    
    if valid_mask.sum() == 0:
        return None, None

    grid_y, grid_x = torch.meshgrid(torch.arange(H_s, device=image.device), 
                                    torch.arange(W_s, device=image.device), 
                                    indexing='ij')
    
    ndc_x = (2.0 * (grid_x + 0.5) / W_s) - 1.0
    ndc_y = (2.0 * (grid_y + 0.5) / H_s) - 1.0
    
    tan_fovx = np.tan(camera.FoVx / 2.0)
    tan_fovy = np.tan(camera.FoVy / 2.0)
    
    z_cam = depth_s.squeeze(0)
    x_cam = z_cam * ndc_x * tan_fovx
    y_cam = z_cam * ndc_y * tan_fovy
    
    xyz_cam_flat = torch.stack([x_cam, y_cam, z_cam], dim=-1).reshape(-1, 3)
    rgb_flat = rgb_s.permute(1, 2, 0).reshape(-1, 3)
    
    xyz_cam_valid = xyz_cam_flat[valid_mask]
    rgb_valid = rgb_flat[valid_mask]
    
    w2c = camera.world_view_transform.transpose(0, 1)
    c2w = torch.inverse(w2c)
    
    ones = torch.ones((xyz_cam_valid.shape[0], 1), device=image.device)
    xyz_cam_homo = torch.cat([xyz_cam_valid, ones], dim=1)
    
    xyz_world_homo = xyz_cam_homo @ c2w.transpose(0, 1)
    xyz_world = xyz_world_homo[:, :3]
    
    return xyz_world, rgb_valid

def compute_reference_ply_embedding(miche_model, ply_path, npoints=4096):
    print(f"Loading reference point cloud from: {ply_path}")
    
    try:
        mesh = trimesh.load(ply_path)
    except Exception as e:
        print(f"Error loading PLY: {e}")
        return None

    vertices = mesh.vertices
    
    colors = None
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[:, :3]
        colors = colors.astype(np.float32) / 255.0
    else:
        print("Warning: Reference PLY has no colors. Using default grey.")
        colors = np.ones_like(vertices) * 0.5

    total_points = vertices.shape[0]
    if total_points >= npoints:
        idx = np.random.choice(total_points, npoints, replace=False)
    else:
        idx = np.arange(total_points)
    
    sampled_xyz = vertices[idx]
    sampled_rgb = colors[idx]

    sampled_xyz = torch.tensor(sampled_xyz, dtype=torch.float32).cuda()
    sampled_rgb = torch.tensor(sampled_rgb, dtype=torch.float32).cuda()

    centroid = torch.mean(sampled_xyz, dim=0, keepdim=True)
    sampled_xyz = sampled_xyz - centroid
    dist = torch.max(torch.sqrt(torch.sum(sampled_xyz ** 2, dim=1)), dim=0)[0]
    sampled_xyz = sampled_xyz / (dist + 1e-6)

    pc = torch.cat([sampled_xyz, sampled_rgb], dim=1)
    pc_input = pc.unsqueeze(0)

    print("Encoding reference point cloud with MICHE...")
    with torch.no_grad():
        ret_dict = miche_model.encode_pc(pc_input)
        
        ref_emb_proj = ret_dict['projected']
        ref_emb_proj = ref_emb_proj / (ref_emb_proj.norm(dim=-1, keepdim=True) + 1e-6)
        
        ref_intermediates = [feat.detach() for feat in ret_dict['intermediates']]
    
    return {
        'projected': ref_emb_proj.detach(),
        'intermediates': ref_intermediates
    }


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    miche_model = None
    reference_pc_embedding = None
    miche_img_transform = get_standard_transform()
    
    use_miche_logic = (opt.lambda_3d_consistency > 0 or opt.lambda_multimodal > 0)
    
    ref_ply_path = getattr(opt, "ref_ply_path", None)

    if use_miche_logic:
        if not ref_ply_path or not os.path.exists(ref_ply_path):
            print(f"Error: MICHE loss enabled but ref_ply_path is invalid: {ref_ply_path}")
            exit()
        else:
            miche_args = SimpleNamespace(
                miche_ckpt_path=getattr(opt, "miche_ckpt_path", None),
                lambda_multimodal=getattr(opt, "lambda_multimodal", 0.0),
                clip_model_version=getattr(opt, "clip_model_version", None)
            )
            miche_model = load_miche_encoder(miche_args)
            
            if miche_model:
                reference_pc_embedding = compute_reference_ply_embedding(
                    miche_model, 
                    ref_ply_path, 
                    npoints=getattr(opt, "miche_npoints", 4096)
                )

    all_cameras = scene.getTrainCameras()
    gt_view_names_set = opt.gt_view_names
    
    gt_cam_list = []
    novel_cam_list = []
    
    for cam in all_cameras:
        if cam.image_name in gt_view_names_set:
            gt_cam_list.append(cam)
        else:
            novel_cam_list.append(cam)
            
    print(f"Split Dataset: {len(gt_cam_list)} GT views, {len(novel_cam_list)} Novel views.")
    if len(gt_cam_list) == 0:
        print("Warning: No GT views found matching the provided names! 'loss_rgb' will be 0.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_3d_cons_for_log = 0.0
    ema_multi_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        loss_rgb = 0.0
        dist_loss = 0.0
        normal_loss = 0.0
        
        if len(gt_cam_list) > 0:
            viewpoint_cam_gt = gt_cam_list[randint(0, len(gt_cam_list)-1)]
            render_pkg_gt = render(viewpoint_cam_gt, gaussians, pipe, background)
            image_gt_rend = render_pkg_gt["render"]
            
            gt_image = viewpoint_cam_gt.original_image.cuda()
            Ll1 = l1_loss(image_gt_rend, gt_image)
            loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_gt_rend, gt_image))
            
            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
            
            if lambda_dist > 0 or lambda_normal > 0:
                rend_dist = render_pkg_gt["rend_dist"]
                rend_normal  = render_pkg_gt['rend_normal']
                surf_normal = render_pkg_gt['surf_normal']
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                
                normal_loss = lambda_normal * (normal_error).mean()
                dist_loss = lambda_dist * (rend_dist).mean()
            
            viewspace_point_tensor = render_pkg_gt["viewspace_points"]
            visibility_filter = render_pkg_gt["visibility_filter"]
            radii = render_pkg_gt["radii"]


        # Multimodal 2D-3D Loss: Rendered Image vs Reference Point Cloud
        loss_multi = 0.0
        
        should_run_multi = (len(novel_cam_list) > 0) and \
                           (miche_model is not None) and \
                           (reference_pc_embedding is not None) and \
                           (iteration > opt.miche_start_iter) and \
                           (opt.lambda_multimodal > 0) and \
                           (iteration % opt.multimodal_interval == 0)

        if should_run_multi:
            viewpoint_cam_novel = novel_cam_list[randint(0, len(novel_cam_list)-1)]
            render_pkg_novel = render(viewpoint_cam_novel, gaussians, pipe, background)
            image_novel_rend = render_pkg_novel["render"]
            
            try:
                # Transform rendered image for CLIP
                img_input = miche_img_transform(image_novel_rend.unsqueeze(0))
                
                # Encode image using MICHE's CLIP
                img_emb_dict = miche_model.encode_image(img_input)
                img_emb = img_emb_dict['projected']
                img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-6)
                
                # Compute similarity with reference point cloud embedding
                similarity_2d_3d = (img_emb @ reference_pc_embedding['projected'].T).squeeze()
                loss_multi = opt.lambda_multimodal * (1.0 - similarity_2d_3d)
                
                if torch.isnan(loss_multi): 
                    loss_multi = 0.0
                
            except Exception as e:
                print(f"Error in multimodal loss: {e}")
                import traceback
                traceback.print_exc()
                loss_multi = 0.0


        # 3D Consistency Loss: Backprojected Point Cloud vs Reference Point Cloud
        loss_3d_cons = 0.0
        
        should_run_3d = (len(gt_cam_list) > 0) and \
                        (miche_model is not None) and \
                        (reference_pc_embedding is not None) and \
                        (iteration > opt.miche_start_iter) and \
                        (opt.lambda_3d_consistency > 0)

        if should_run_3d:
            try:
                all_xyz = []
                all_rgb = []
                
                for cam_for_3d in gt_cam_list:
                    render_pkg_3d = render(cam_for_3d, gaussians, pipe, background)
                    image_for_3d = render_pkg_3d["render"]
                    depth_for_3d = render_pkg_3d["surf_depth"]
                    
                    xyz_world, rgb_points = get_masked_backprojected_pc(
                        image_for_3d, 
                        depth_for_3d, 
                        cam_for_3d, 
                        downsample_factor=8,
                        mask_threshold=0.5
                    )
                    
                    if xyz_world is not None and xyz_world.shape[0] > 10:
                        all_xyz.append(xyz_world)
                        all_rgb.append(rgb_points)
                
                if len(all_xyz) > 0:
                    merged_xyz = torch.cat(all_xyz, dim=0)
                    merged_rgb = torch.cat(all_rgb, dim=0)
                    
                    centroid = torch.mean(merged_xyz, dim=0, keepdim=True)
                    xyz_norm = merged_xyz - centroid
                    dist = torch.max(torch.sqrt(torch.sum(xyz_norm ** 2, dim=1)), dim=0)[0]
                    xyz_norm = xyz_norm / (dist + 1e-6)
                    
                    target_n = getattr(opt, "miche_npoints", 4096)
                    curr_n = xyz_norm.shape[0]
                    
                    if curr_n >= target_n:
                        idx = torch.randperm(curr_n, device="cuda")[:target_n]
                    else:
                        idx = torch.arange(curr_n, device="cuda")
                        
                    sampled_xyz = xyz_norm[idx]
                    sampled_rgb = merged_rgb[idx]
                    
                    sampled_rgb = torch.clamp(sampled_rgb, 0.0, 1.0)
                    
                    pc_input = torch.cat([sampled_xyz, sampled_rgb], dim=1).unsqueeze(0)

                    curr_ret_dict = miche_model.encode_pc(pc_input)
                    curr_intermediates = curr_ret_dict['intermediates']
                    ref_intermediates = reference_pc_embedding['intermediates']
                    
                    loss_per_layer = 0.0
                    num_layers = len(curr_intermediates)
                    
                    for i in range(num_layers):
                        feat_curr = curr_intermediates[i]
                        feat_ref = ref_intermediates[i]
                        
                        if opt.loss_3d_type == 'cosine':
                            feat_curr_norm = feat_curr / (feat_curr.norm(dim=-1, keepdim=True) + 1e-6)
                            feat_ref_norm = feat_ref / (feat_ref.norm(dim=-1, keepdim=True) + 1e-6)
                            layer_loss = 1.0 - (feat_curr_norm.flatten(1) @ feat_ref_norm.flatten(1).T).mean()
                        elif opt.loss_3d_type == 'mse':
                            layer_loss = F.mse_loss(feat_curr, feat_ref)
                        else:
                            layer_loss = 0.0
                        loss_per_layer += layer_loss

                    loss_3d_cons = opt.lambda_3d_consistency * (loss_per_layer / num_layers)
                else:
                    loss_3d_cons = 0.0
                    
                if torch.isnan(loss_3d_cons): 
                    loss_3d_cons = 0.0
                
            except Exception as e:
                print(f"Error in 3D cons (Backproj): {e}")
                import traceback
                traceback.print_exc()
                loss_3d_cons = 0.0

        total_loss = loss_rgb + dist_loss + normal_loss + loss_multi + loss_3d_cons
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            if len(gt_cam_list) > 0 and iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            l_rgb_val = loss_rgb.item() if isinstance(loss_rgb, torch.Tensor) else loss_rgb
            l_multi_val = loss_multi.item() if isinstance(loss_multi, torch.Tensor) else loss_multi
            l_3d_val = loss_3d_cons.item() if isinstance(loss_3d_cons, torch.Tensor) else loss_3d_cons
            
            ema_loss_for_log = 0.4 * l_rgb_val + 0.6 * ema_loss_for_log
            ema_multi_for_log = 0.4 * l_multi_val + 0.6 * ema_multi_for_log
            ema_3d_cons_for_log = 0.4 * l_3d_val + 0.6 * ema_3d_cons_for_log
            
            if iteration % 10 == 0:
                loss_dict = {
                    "L_RGB": f"{ema_loss_for_log:.{4}f}",
                    "L_Multi": f"{ema_multi_for_log:.{4}f}",
                    "L_3D": f"{ema_3d_cons_for_log:.{4}f}",
                    "Pts": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/loss_rgb', ema_loss_for_log, iteration)
                if iteration > opt.miche_start_iter:
                    tb_writer.add_scalar('train_loss_patches/loss_multimodal', ema_multi_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/loss_3d_consistency', ema_3d_cons_for_log, iteration)
                
            training_report(
                tb_writer, 
                iteration, 
                Ll1 if len(gt_cam_list) > 0 else torch.tensor(0.0), 
                total_loss, 
                l1_loss, 
                iter_start.elapsed_time(iter_end), 
                testing_iterations, 
                scene, 
                render, 
                (pipe, background),
                gt_view_names_set
            )
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

                if len(gt_cam_list) > 0:
                    try:
                        all_xyz_debug = []
                        all_rgb_debug = []
                        pc_save_path = os.path.join(scene.model_path, f"point_cloud/iteration_{iteration}")
                        os.makedirs(pc_save_path, exist_ok=True)
                        
                        for debug_cam in gt_cam_list:
                            render_pkg_debug = render(debug_cam, gaussians, pipe, background)
                            
                            xyz_world_debug, rgb_debug = get_masked_backprojected_pc(
                                render_pkg_debug["render"], 
                                render_pkg_debug["surf_depth"], 
                                debug_cam, 
                                downsample_factor=8,
                                mask_threshold=0.5
                            )
                            
                            if xyz_world_debug is not None:
                                all_xyz_debug.append(xyz_world_debug)
                                all_rgb_debug.append(rgb_debug)
                                
                                pts = xyz_world_debug.detach().cpu().numpy()
                                clrs = (rgb_debug.detach().cpu().numpy() * 255).astype(np.uint8)
                                
                                pcd = trimesh.PointCloud(vertices=pts, colors=clrs)
                                ply_name = f"backproj_masked_{debug_cam.image_name}.ply"
                                pcd.export(os.path.join(pc_save_path, ply_name))
                                print(f"Saved masked back-projected PC to {os.path.join(pc_save_path, ply_name)}")
                        
                        if len(all_xyz_debug) > 0:
                            merged_xyz_debug = torch.cat(all_xyz_debug, dim=0)
                            merged_rgb_debug = torch.cat(all_rgb_debug, dim=0)
                            
                            pts_merged = merged_xyz_debug.detach().cpu().numpy()
                            clrs_merged = (merged_rgb_debug.detach().cpu().numpy() * 255).astype(np.uint8)
                            
                            pcd_merged = trimesh.PointCloud(vertices=pts_merged, colors=clrs_merged)
                            ply_name_merged = "backproj_masked_merged.ply"
                            pcd_merged.export(os.path.join(pc_save_path, ply_name_merged))
                            print(f"Saved merged masked back-projected PC to {os.path.join(pc_save_path, ply_name_merged)}")
                        else:
                            print("Warning: No points survived mask filtering during debug save.")
                        
                    except Exception as e:
                        print(f"Failed to save debug point cloud: {e}")
            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    progress_bar.close()
    print("\nTraining complete.")

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

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, gt_view_names_set):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        
        sparse_input_cams = [c for c in train_cameras if c.image_name in gt_view_names_set]
        held_out_cams = [c for c in train_cameras if c.image_name not in gt_view_names_set]

        if len(held_out_cams) > 0:
            held_out_cams = held_out_cams[::max(1, len(held_out_cams)//5)]

        validation_configs = [
            {'name': 'test', 'cameras': test_cameras},
            {'name': 'train_sparse_gt', 'cameras': sparse_input_cams},
            {'name': 'train_novel_view', 'cameras': held_out_cams}
        ]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 3):
                        from utils.general_utils import colormap
                        
                        prefix = config['name'] + "_view_{}".format(viewpoint.image_name)
                        tb_writer.add_images(prefix + "/render", image[None], global_step=iteration)
                        
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(prefix + "/ground_truth", gt_image[None], global_step=iteration)

                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / (norm + 1e-6)
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(prefix + "/depth", depth[None], global_step=iteration)

                        try:
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(prefix + "/surf_normal", surf_normal[None], global_step=iteration)
                            rend_alpha = render_pkg['rend_alpha']
                            tb_writer.add_images(prefix + "/rend_alpha", rend_alpha[None], global_step=iteration)
                        except:
                            pass

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                
                print("\n[ITER {}] Evaluating {}: L1 {:.6f} PSNR {:.6f}".format(iteration, config['name'], l1_test, psnr_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters (MICHE version with 2D-3D multimodal loss)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--lambda_3d_consistency", type=float, default=0.1, help="Weight for GS-Point to Ref-Point loss")
    parser.add_argument("--loss_3d_type", type=str, default="mse", choices=["cosine", "mse"], help="Type of 3D consistency loss: cosine or mse")
    parser.add_argument("--lambda_multimodal", type=float, default=0.1, help="Weight for Render-Image to Ref-Point loss (using MICHE CLIP)")
    parser.add_argument("--multimodal_interval", type=int, default=1, help="Interval for computing multimodal loss")
    parser.add_argument("--gt_view_names", nargs="+", type=str, default=[], help="List of image names to use ground truth L1/SSIM loss")
    
    parser.add_argument("--miche_ckpt_path", type=str, default="shapevae-256.ckpt", help="Path to MICHE checkpoint")
    parser.add_argument("--miche_npoints", type=int, default=20000)
    parser.add_argument("--miche_start_iter", type=int, default=7000)
    parser.add_argument("--ref_ply_path", type=str, default="", help="Path to the reference .ply")
    parser.add_argument("--clip_model_version", type=str, default=None, help="CLIP model version for image encoding")

    args = parser.parse_args(sys.argv[1:])

    print("test_iterations", args.test_iterations)
    print("save_iterations", args.save_iterations)

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    opt_params = op.extract(args)
    opt_params.lambda_3d_consistency = args.lambda_3d_consistency
    opt_params.loss_3d_type = args.loss_3d_type
    opt_params.lambda_multimodal = args.lambda_multimodal
    opt_params.multimodal_interval = args.multimodal_interval
    opt_params.gt_view_names = set(args.gt_view_names) if args.gt_view_names else set()
    opt_params.miche_ckpt_path = args.miche_ckpt_path
    opt_params.miche_npoints = args.miche_npoints
    opt_params.miche_start_iter = args.miche_start_iter
    opt_params.ref_ply_path = args.ref_ply_path
    opt_params.clip_model_version = args.clip_model_version

    print("opt_params.lambda_3d_consistency", opt_params.lambda_3d_consistency)
    print("opt_params.lambda_multimodal", opt_params.lambda_multimodal)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), opt_params, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    print("\nTraining complete.")
