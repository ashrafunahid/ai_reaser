import os
import subprocess
from PIL import Image
import uuid

def pad_to_modulo8(image: Image.Image, target_size=None):
    w, h = image.size
    if target_size:
        new_w, new_h = target_size
    else:
        new_w = ((w + 7) // 8) * 8
        new_h = ((h + 7) // 8) * 8
    padded = Image.new(image.mode, (new_w, new_h))
    padded.paste(image, (0, 0))
    return padded, (new_w, new_h)

def run_zitspp(image_path, mask_path, device='cuda'):
    base_output_dir = 'media/outputs/zitspp'
    os.makedirs(base_output_dir, exist_ok=True)

    # Create a temp folder
    temp_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(base_output_dir, f"temp_{temp_id}")
    os.makedirs(temp_dir, exist_ok=True)

    # Step 1: Load and pad
    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    padded_img, target_size = pad_to_modulo8(img)
    padded_mask, _ = pad_to_modulo8(mask, target_size=target_size)

    # Step 2: Save into temp_dir
    padded_img_path = os.path.join(temp_dir, 'input.png')
    padded_mask_path = os.path.join(temp_dir, 'mask.png')
    padded_img.save(padded_img_path)
    padded_mask.save(padded_mask_path)

    # Step 3: Run subprocess with temp_dir as img_dir and mask_dir
    cmd = [
        'python', 'ZITS_PlusPlus/test.py',
        '--config', 'ZITS_PlusPlus/configs/config_zitspp_finetune.yml',
        '--ckpt_resume', 'ZITS_PlusPlus/ckpts/model_512/models/last.ckpt',
        '--wf_ckpt', 'ZITS_PlusPlus/ckpts/model_512/models/best_lsm_hawp.pth', 
        '--save_path', base_output_dir,
        '--img_dir', temp_dir,
        '--mask_dir', temp_dir,
        '--use_ema',
        '--exp_name', f'zitspp_{temp_id}'
    ]

    subprocess.run(cmd, check=True)

    # Step 4: Look for output inpainted image (not input.png!)
    result_dir = os.path.join(base_output_dir, f'zitspp_{temp_id}')
    if not os.path.exists(result_dir):
        print(f"[ERROR] Result directory not found: {result_dir}")
        return None

    # Look for an image that starts with "inpainted"
    inpainted_file = None
    for file in os.listdir(result_dir):
        if file.startswith("inpainted") and file.endswith((".png", ".jpg")):
            inpainted_file = os.path.join(result_dir, file)
            break

    if not inpainted_file or not os.path.exists(inpainted_file):
        print(f"[ERROR] Inpainted output not found in {result_dir}")
        return None

    return inpainted_file


