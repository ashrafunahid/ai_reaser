import os
import subprocess

def run_zitspp(image_path, mask_path, device='cuda'):
    output_dir = 'media/outputs/zitspp'
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'python', 'ZITS_PlusPlus/test.py',
        '--config', 'ZITS_PlusPlus/configs/config_zitspp_finetune.yml',
        '--ckpt_resume', 'ZITS_PlusPlus/ckpts/model_512/models/last.ckpt',
        '--save_path', output_dir,
        '--img_dir', os.path.dirname(image_path),
        '--mask_dir', os.path.dirname(mask_path),
        '--use_ema',
        '--object_removal',
    ]

    print(cmd)
    # >>> Debugging:
    for item in cmd:
        if isinstance(item, tuple):
            raise ValueError(f"Tuple found in cmd list: {item}")

    subprocess.run(cmd, check=True)
    return os.path.join(output_dir, os.path.basename(image_path))
