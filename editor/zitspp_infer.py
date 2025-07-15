import sys
import torch
import cv2
from omegaconf import OmegaConf
from torchvision import transforms
from PIL import Image
from pathlib import Path

ZITSPP_BASE = Path(__file__).resolve().parent.parent / "ZITS-PlusPlus"
sys.path.append(str(ZITSPP_BASE))

from networks.trainer import Trainer

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def load_mask(mask_path, size):
    return Image.open(mask_path).convert("L").resize(size)

def run_zitspp(image_path, mask_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config_path = ZITSPP_BASE / "configs/config_zitspp_finetune.yml"
    config = OmegaConf.load(config_path)

    # Load model
    model = Trainer(config)
    model_path = ZITSPP_BASE / "ckpts/models/last.ckpt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    # Load image and mask
    image = load_image(image_path)
    mask = load_mask(mask_path, image.size)

    # Transform
    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)
    mask_tensor = transform(mask).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output, _ = model.forward_a(img_tensor, mask_tensor)

    # Post-process and save
    result = output[0].cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255
    result = result.astype('uint8')
    return result
