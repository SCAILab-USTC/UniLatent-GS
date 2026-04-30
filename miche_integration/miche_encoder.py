import sys
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from miche_integration.miche.encode import load_model as load_miche_core_model
from miche_integration.miche.michelangelo.utils.misc import instantiate_from_config

class MICHEEncoder:
    def __init__(self, ckpt_path=None, config_path=None, enable_clip_image=False, clip_model_version=None):
        self.model = None
        self.ckpt_path = ckpt_path
        self.enable_clip_image = enable_clip_image
        self.clip_model_version = clip_model_version
        self.config_clip_version = None  # Will be loaded from config file
        
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "shapevae-256_no_flash.yaml"
            )
        self.config_path = config_path
        
    def load_model(self):
        print(f"Loading MICHE Model from config: {self.config_path}")
        
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            model_config = OmegaConf.load(self.config_path)
            
            # Extract clip_model_version from config if available
            if hasattr(model_config, "model"):
                model_params = model_config.model
            else:
                model_params = model_config
            
            if hasattr(model_params, "params") and hasattr(model_params.params, "aligned_module_cfg"):
                aligned_cfg = model_params.params.aligned_module_cfg
                if hasattr(aligned_cfg, "params") and hasattr(aligned_cfg.params, "clip_model_version"):
                    self.config_clip_version = aligned_cfg.params.clip_model_version
                    print(f"Found clip_model_version in config: {self.config_clip_version}")
            
            if hasattr(model_config, "model"):
                model_config = model_config.model
            
            if self.ckpt_path and os.path.exists(self.ckpt_path):
                print(f"Using checkpoint: {self.ckpt_path}")
                self.model = instantiate_from_config(model_config, ckpt_path=self.ckpt_path)
            else:
                print("No valid checkpoint provided. Loading without checkpoint (random init).")
                self.model = instantiate_from_config(model_config, ckpt_path=None)
            
            self.model = self.model.cuda()
            self.model = self.model.eval()
            
            # Enable CLIP image encoding if requested
            if self.enable_clip_image:
                self._enable_clip_model()
            
            print("MICHE model loaded successfully (flash attention disabled)")
            return self
        except Exception as e:
            print(f"Failed to load MICHE model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _enable_clip_model(self):
        """Enable CLIP model for image encoding"""
        try:
            from transformers import CLIPModel
            
            # Priority: 1) Explicitly provided version, 2) Config file version, 3) Default
            if self.clip_model_version is not None:
                clip_version = self.clip_model_version
                print(f"Using explicitly provided CLIP model version: {clip_version}")
            elif self.config_clip_version is not None:
                clip_version = self.config_clip_version
                print(f"Using CLIP model version from config: {clip_version}")
            else:
                clip_version = "openai/clip-vit-large-patch14"
                print(f"Using default CLIP model version: {clip_version}")
            
            print(f"Loading CLIP model for image encoding: {clip_version}")
            self.model.model.clip_model = CLIPModel.from_pretrained(clip_version)
            self.model.model.clip_model = self.model.model.clip_model.cuda()
            self.model.model.clip_model.eval()
            for param in self.model.model.clip_model.parameters():
                param.requires_grad = False
            print("CLIP model loaded successfully for image encoding")
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            self.enable_clip_image = False
            
    def encode_pc(self, pc):
        """
        Encode point cloud using MICHE
        
        Args:
            pc: Point cloud of shape [B, N, 6] (xyz + rgb) or [B, N, 3] (xyz only)
            
        Returns:
            dict with projected embedding and intermediate features
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        self.model.eval()
        
        with torch.no_grad():
            batch_size = pc.shape[0]
            n_points = pc.shape[1]
            
            pc_xyz = pc[:, :, :3]
            
            if pc.shape[2] >= 6:
                pc_rgb = pc[:, :, 3:6]
            else:
                pc_rgb = torch.ones_like(pc_xyz) * 0.5
            
            pc_normal = torch.zeros_like(pc_xyz)
            
            surface = torch.cat([pc_xyz, pc_normal], dim=-1)
            
            shape_embed, shape_latents = self.model.model.encode_shape_embed(surface, return_latents=True)
            
            projected = shape_embed
            intermediates = [shape_latents]
            
            return {
                'projected': projected,
                'intermediates': intermediates
            }
            
    def encode_image(self, img):
        """
        Encode image using MICHE's CLIP model
        
        Args:
            img: Image tensor of shape [B, 3, H, W] with values in [0, 1]
            
        Returns:
            dict with projected embedding
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not self.enable_clip_image or self.model.model.clip_model is None:
            raise RuntimeError("CLIP image encoding not enabled. Initialize with enable_clip_image=True")
            
        self.model.eval()
        
        with torch.no_grad():
            # Encode image using CLIP
            image_embed = self.model.model.encode_image_embed(img)
            
            # Project to shape embedding space
            projected = image_embed @ self.model.model.shape_projection
            
            return {
                'projected': projected
            }
        
def load_miche_model(ckpt_path=None, enable_clip_image=False):
    """
    Helper function to load MICHE encoder
    
    Args:
        ckpt_path: Path to MICHE checkpoint
        enable_clip_image: Whether to enable CLIP image encoding
        
    Returns:
        MICHEEncoder instance
    """
    encoder = MICHEEncoder(ckpt_path=ckpt_path, enable_clip_image=enable_clip_image)
    return encoder.load_model()
