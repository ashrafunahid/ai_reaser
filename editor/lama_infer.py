import os
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.trainers import load_checkpoint

# Fix for PyTorch 2.6+ unpickling error
from torch.serialization import add_safe_globals
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from omegaconf.dictconfig import DictConfig
from omegaconf.base import ContainerMetadata
add_safe_globals({ModelCheckpoint, DictConfig, ContainerMetadata})

def load_lama_model(model_path='lama_saic/models/big-lama'):
    config_path = os.path.join(model_path, 'config.yaml')
    checkpoint_path = os.path.join(model_path, 'models', 'best.ckpt')

    config = OmegaConf.load(config_path)
    config.training_model.predict_only = True
    config.visualizer.kind = 'noop'

    model = load_checkpoint(config, checkpoint_path, strict=False)
    model.eval()
    return model


def run_lama_inpainting(image_path, mask_path, model):
    # Load Image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert in RGB
    image = image.astype(np.float32) / 255.0  # [0, 1] range
    
    # Load mask and preprocess
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.float32)  # Binary mask [0, 1]
    
    # Resize for dimension matching
    target_size = (image.shape[1], image.shape[0])  # (width, height)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Resize into 32x
    def to_32s(x):
        return int(round(x / 32) * 32)
    
    h, w = image.shape[:2]
    new_h, new_w = to_32s(h), to_32s(w)
    
    if (new_h != h) or (new_w != w):
        image = cv2.resize(image, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Make Batch Dictionary
    batch = {
        'image': torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0),  # [1, 3, H, W]
        'mask': torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
    }
    
    # Move to Device
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Prediction
    with torch.no_grad():
        result = model(batch)
    
    # Process Output
    inpainted = result['inpainted'][0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    inpainted = (inpainted * 255).clip(0, 255).astype(np.uint8)  # [0, 255] reange
    
    return inpainted