import os
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig # Ensure DictConfig is imported
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.trainers import load_checkpoint

# Fix for PyTorch 2.6+ unpickling error
from torch.serialization import add_safe_globals
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo # Import for padding
# from omegaconf.dictconfig import DictConfig # Already imported above
from omegaconf.base import ContainerMetadata
add_safe_globals({ModelCheckpoint, DictConfig, ContainerMetadata})

def load_lama_model(model_path='lama_saic/models/big-lama'):
    model_config_path = os.path.join(model_path, 'config.yaml')
    generic_prediction_config_path = 'lama_saic/configs/prediction/default.yaml'
    checkpoint_path = os.path.join(model_path, 'models', 'best.ckpt')

    if not os.path.exists(model_config_path):
        print(f"CRITICAL: Model config file not found at {model_config_path}")
        # This should ideally raise an error. For now, OmegaConf.load will fail later if path is truly bad.
        pass

    config = OmegaConf.load(model_config_path)
    config.training_model.predict_only = True
    config.visualizer.kind = 'noop'

    model = load_checkpoint(config, checkpoint_path, strict=False)
    model.eval()

    # --- Determine refiner parameters and run flag ---
    run_refiner_flag = True
    refiner_params = {} # Initialize

    # Default gpu_ids based on actual hardware, prioritizing single GPU for refiner
    default_gpu_ids_for_refiner = ""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            default_gpu_ids_for_refiner = "0," # Default to using only the first GPU for refine_predict
        print(f"DEBUG: CUDA is available. Number of GPUs: {num_gpus}. Initial default_gpu_ids_for_refiner: '{default_gpu_ids_for_refiner}'")

    if not os.path.exists(generic_prediction_config_path):
        print(f"WARNING: Generic prediction config for refiner not found at {generic_prediction_config_path}. Using hardcoded failsafe refiner parameters.")
        refiner_params = {
            'gpu_ids': default_gpu_ids_for_refiner,
            'modulo': 8, 'n_iters': 10, 'lr': 0.002,
            'min_side': 256, 'max_scales': 3, 'px_budget': 500000
        }
    else:
        pred_config = OmegaConf.load(generic_prediction_config_path)
        print(f"DEBUG: Forcing 'run_refiner_flag' to True to attempt refinement (overriding {generic_prediction_config_path} 'refine' key if present).")

        default_modulo = pred_config.get('dataset', {}).get('pad_out_to_modulo', 8)
        refiner_params_conf = pred_config.get('refiner', {})

        configured_gpu_ids = refiner_params_conf.get('gpu_ids', default_gpu_ids_for_refiner)

        current_gpu_ids_for_refiner = default_gpu_ids_for_refiner
        if torch.cuda.is_available():
            if ',' in configured_gpu_ids and configured_gpu_ids != default_gpu_ids_for_refiner:
                print(f"WARNING: Configured gpu_ids ('{configured_gpu_ids}') in {generic_prediction_config_path} suggests multi-GPU for refiner. Overriding to use single GPU ('{default_gpu_ids_for_refiner}') for stability.")
                current_gpu_ids_for_refiner = default_gpu_ids_for_refiner
            elif configured_gpu_ids:
                current_gpu_ids_for_refiner = configured_gpu_ids if configured_gpu_ids.endswith(',') else configured_gpu_ids + ","
        else:
            current_gpu_ids_for_refiner = ""

        refiner_params = {
            'gpu_ids': current_gpu_ids_for_refiner,
            'modulo': refiner_params_conf.get('modulo', default_modulo),
            'n_iters': refiner_params_conf.get('n_iters', 15),
            'lr': refiner_params_conf.get('lr', 0.002),
            'min_side': refiner_params_conf.get('min_side', 512),
            'max_scales': refiner_params_conf.get('max_scales', 3),
            'px_budget': refiner_params_conf.get('px_budget', 1800000)
        }
        print(f"DEBUG: Loaded/determined refiner parameters: {refiner_params}")

    # --- Consistently move model to a target device ---
    target_device_str = "cpu"
    if torch.cuda.is_available():
        gpu_ids_for_main_model = refiner_params.get('gpu_ids', "")
        if gpu_ids_for_main_model:
            try:
                first_gpu_id = gpu_ids_for_main_model.split(',')[0].strip()
                if first_gpu_id:
                     target_device_str = f"cuda:{first_gpu_id}"
                else: # If first_gpu_id is empty string (e.g. gpu_ids_for_main_model was just ",")
                     target_device_str = "cuda:0" # Default to cuda:0
            except Exception as e:
                 print(f"Warning: Could not parse GPU ID for main model from refiner_params['gpu_ids'] ('{gpu_ids_for_main_model}'). Defaulting to cuda:0. Error: {e}")
                 target_device_str = "cuda:0"
        elif target_device_str == "cpu": # CUDA is available, but gpu_ids_for_main_model is empty
            print("DEBUG: CUDA available, but refiner gpu_ids is empty. Defaulting main model to cuda:0.")
            target_device_str = "cuda:0"
            if not refiner_params.get('gpu_ids'): refiner_params['gpu_ids'] = "0,"


    target_device = torch.device(target_device_str)
    try:
        model.to(target_device)
        print(f"DEBUG: LaMa model explicitly moved to device: {target_device}")
    except Exception as e:
        print(f"ERROR: Failed to move LaMa model to device {target_device}. Error: {e}. Model will remain on CPU or its initial device.")
        target_device = next(model.parameters()).device
        print(f"DEBUG: Model is currently on device: {target_device}")


    final_refiner_config = {'run_refiner': run_refiner_flag, 'params': refiner_params}
    if target_device.type == 'cpu':
        final_refiner_config['params']['gpu_ids'] = ""
        print("DEBUG: Model is on CPU, ensuring refiner_params['gpu_ids'] is empty.")
    elif target_device.type == 'cuda' and refiner_params.get('gpu_ids'):
        # Ensure gpu_ids for refiner matches the model's device index if it's a single GPU setup
        # This logic assumes single GPU for refiner if model is on GPU.
        current_refiner_gpu_ids = final_refiner_config['params']['gpu_ids']
        # Check if it's not already correctly set to a single device (e.g. "0,")
        if not (str(target_device.index) in current_refiner_gpu_ids.split(',') and not any(id_ for id_ in current_refiner_gpu_ids.split(',') if id_ and id_ != str(target_device.index))):
             new_gpu_ids = f"{target_device.index},"
             print(f"DEBUG: Model on {target_device}, ensuring refiner_params['gpu_ids'] is '{new_gpu_ids}' for single GPU operation.")
             final_refiner_config['params']['gpu_ids'] = new_gpu_ids


    print(f"DEBUG: Final refiner configuration to be used: {final_refiner_config}")

    return model, final_refiner_config


def run_lama_inpainting(image_path, mask_path, model, refiner_config):
    # Load Image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float_np = image.astype(np.float32) / 255.0

    # Load mask and preprocess
    mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_float_np = (mask_cv > 127).astype(np.float32)
    
    # Resize mask to match original image dimensions first
    target_size_orig = (image.shape[1], image.shape[0])  # (width, height)
    mask_resized_np = cv2.resize(mask_float_np, target_size_orig, interpolation=cv2.INTER_NEAREST)
    
    h_orig, w_orig = image.shape[:2]

    # --- Convert to Tensors ---
    image_tensor = torch.from_numpy(image_float_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask_tensor = torch.from_numpy(mask_resized_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # --- Pad Tensors for Model Input ---
    modulo = refiner_config.get('params', {}).get('modulo', 8)
    print(f"DEBUG: Using modulo {modulo} for padding.")

    padded_image_tensor = pad_tensor_to_modulo(image_tensor, modulo)
    padded_mask_tensor = pad_tensor_to_modulo(mask_tensor, modulo)
    
    # --- Create Batch ---
    batch = {
        'image': padded_image_tensor,
        'mask': padded_mask_tensor,
        'unpad_to_size': (h_orig, w_orig) # Use Python tuple
    }
    
    device = next(model.parameters()).device
    batch_on_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    batch_on_device['unpad_to_size'] = batch['unpad_to_size']

    with torch.no_grad():
        prediction_result_dict = model(batch_on_device)
        inpainted_tensor_padded = prediction_result_dict['inpainted']

        if refiner_config['run_refiner'] and refiner_config['params'].get('gpu_ids', "") != "": # Also check if gpu_ids is not empty for refiner
            print(f"DEBUG: Applying refinement with params: {refiner_config['params']}")
            try:
                refined_output_tensor_cropped = refine_predict(batch_on_device, model, **refiner_config['params'])
                final_processed_tensor = refined_output_tensor_cropped
                print("DEBUG: Refinement applied successfully. Using refined_tensor (already cropped).")
            except Exception as e:
                print(f"ERROR during refine_predict: {e}")
                import traceback
                traceback.print_exc()
                print("DEBUG: Falling back to non-refined output due to error.")
                h_pad, w_pad = inpainted_tensor_padded.shape[-2:]
                if h_pad > h_orig or w_pad > w_orig :
                    final_processed_tensor = inpainted_tensor_padded[..., :h_orig, :w_orig]
                else:
                    final_processed_tensor = inpainted_tensor_padded
                print(f"DEBUG: Using non-refined output, cropped from padded size {inpainted_tensor_padded.shape} to {final_processed_tensor.shape}")
        else:
            if not refiner_config['run_refiner']:
                 print("DEBUG: Refinement skipped as per 'run_refiner_flag' = False.")
            elif refiner_config['params'].get('gpu_ids', "") == "":
                 print("DEBUG: Refinement skipped as 'gpu_ids' is empty (likely CPU mode).")

            print("DEBUG: Cropping model output directly.")
            h_pad, w_pad = inpainted_tensor_padded.shape[-2:]
            if h_pad > h_orig or w_pad > w_orig :
                 final_processed_tensor = inpainted_tensor_padded[..., :h_orig, :w_orig]
            else:
                 final_processed_tensor = inpainted_tensor_padded
            print(f"DEBUG: Using non-refined output, cropped from padded size {inpainted_tensor_padded.shape} to {final_processed_tensor.shape}")

    output_image_np = final_processed_tensor[0].permute(1, 2, 0).cpu().numpy()
    
    if output_image_np.shape[0] != h_orig or output_image_np.shape[1] != w_orig:
        print(f"WARNING: Output shape {output_image_np.shape[:2]} still mismatching original {(h_orig, w_orig)}. Resizing forcefully.")
        output_image_np = cv2.resize(output_image_np, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)

    final_image_uint8 = (output_image_np * 255).clip(0, 255).astype(np.uint8)
    
    return final_image_uint8