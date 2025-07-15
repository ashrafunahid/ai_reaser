import os
import sys
from pathlib import Path
import torch
import cv2
from omegaconf import OmegaConf
from torchvision import transforms
from PIL import Image

# Add ZITS_inpainting/src to sys.path
ZITS_BASE = Path(__file__).resolve().parent.parent / "ZITS_inpainting"
ZITS_SRC = ZITS_BASE / "src"
sys.path.append(str(ZITS_SRC))

# Import from correct model path
from models.network_zits import InpaintingModel


def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def load_mask(mask_path, size):
    return Image.open(mask_path).convert("L").resize(size)

def run_zits(image_path, mask_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = ZITS_BASE / "config_list/config_ZITS_HR_places2.yml"
    config = OmegaConf.load(config_path)

    # Load model
    model_path = ZITS_BASE / "ckpt/zits_places2_hr/InpaintingModel_best_gen.pth"
    model = InpaintingModel(config)  # <-- You missed instantiating the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    # Load image and mask
    image = load_image(image_path)
    mask = load_mask(mask_path, image.size)

    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)
    mask_tensor = transform(mask).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor, mask_tensor)

    result = output[0].cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255
    result = result.astype('uint8')
    return result
