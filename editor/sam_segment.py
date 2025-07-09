import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import torch
import torch.nn.functional as F

# Load model
sam_checkpoint = "segment-anything/checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def generate_mask_with_point(image_path, input_point):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    input_point_np = np.array([input_point])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point_np,
        point_labels=input_label,
        multimask_output=False,
    )
    return masks[0]



import os # JULES: Added for path manipulation

def generate_mask_with_mask(image_path, mask_array):
    # --- JULES: Debugging ---
    image_id_str = os.path.splitext(os.path.basename(image_path))[0] # Extract image id assuming format like '1.png' or 'some_name.jpg'
    debug_mask_dir = os.path.join('media', 'debug')
    os.makedirs(debug_mask_dir, exist_ok=True)

    print(f"[SAM Debug {image_id_str}] generate_mask_with_mask called.")
    print(f"[SAM Debug {image_id_str}] Input mask_array - Shape: {mask_array.shape}, Dtype: {mask_array.dtype}, Unique values: {np.unique(mask_array)}")
    cv2.imwrite(os.path.join(debug_mask_dir, f"sam_input_mask_{image_id_str}.png"), mask_array * 255) # Renamed temp_debug_mask
    # --- END JULES ---

    # Load and prepare the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]  # (height, width)
    print(f"[SAM Debug {image_id_str}] Original image size: {original_size}")

    # Set image in predictor
    predictor.set_image(image)

    # Prepare mask
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]  # Use first channel if RGB
        print(f"[SAM Debug {image_id_str}] Mask array reshaped from 3D to 2D.")

    # Convert to binary float32 mask
    # The input mask_array from views.py is already 0s and 1s (uint8).
    # We just need to convert it to float32.
    mask_array = mask_array.astype(np.float32)
    print(f"[SAM Debug {image_id_str}] Mask array converted to float32. Shape: {mask_array.shape}, Dtype: {mask_array.dtype}, Unique values: {np.unique(mask_array)}")


    with torch.no_grad():
        # Get image embeddings
        image_embedding = predictor.get_image_embedding()
        _, _, h_feat, w_feat = image_embedding.shape
        print(f"[SAM Debug {image_id_str}] Image embedding feature map size: (h_feat={h_feat}, w_feat={w_feat})")

        # Resize drawn mask to match embedding feature map
        mask_resized = cv2.resize(mask_array, (w_feat, h_feat), interpolation=cv2.INTER_LINEAR)
        print(f"[SAM Debug {image_id_str}] Mask resized (float) to feature map size. Shape: {mask_resized.shape}, Dtype: {mask_resized.dtype}, Unique values: {np.unique(mask_resized)}")

        # --- JULES: Binarize the resized mask before using as prompt ---
        mask_resized = (mask_resized > 0.5).astype(np.float32) # JULES: Reverted: Binarize the coarse mask
        print(f"[SAM Debug {image_id_str}] Binarized mask_resized. Shape: {mask_resized.shape}, Dtype: {mask_resized.dtype}, Unique values: {np.unique(mask_resized)}")
        # --- END JULES ---
        cv2.imwrite(os.path.join(debug_mask_dir, f"sam_mask_resized_binarized_prompt_{image_id_str}.png"), (mask_resized * 255).astype(np.uint8)) # Save the binarized version for inspection
        # --- END JULES ---

        # Convert to tensor
        dense_prompt = torch.as_tensor(mask_resized, dtype=torch.float32, device=device)
        dense_prompt = dense_prompt.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, h, w]
        print(f"[SAM Debug {image_id_str}] Dense_prompt tensor (from binarized mask_resized) - Shape: {dense_prompt.shape}, Dtype: {dense_prompt.dtype}, Min: {dense_prompt.min()}, Max: {dense_prompt.max()}")
        # --- END JULES ---

        # Get dense positional encodings
        image_pe = predictor.model.prompt_encoder.get_dense_pe()

        # Sparse prompt is empty
        sparse_embeddings = torch.zeros(
            (1, 0, predictor.model.prompt_encoder.embed_dim),
            device=device
        )

        # Decode mask
        # --- JULES: Implement multimask_output=True ---
        outputs = predictor.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False
        )

        masks = outputs[0] if isinstance(outputs, tuple) else outputs  # Handle old/new API
        mask_output = masks[0, 0].cpu().numpy()
        mask_output = (mask_output > 0).astype(np.uint8)

        # Resize output mask to original image size
        mask_output = cv2.resize(mask_output, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    return mask_output


# def generate_mask_with_mask(image_path, mask_array):
#     # Read and prepare image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image at {image_path}")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     original_size = image.shape[:2]  # Store original image dimensions
    
#     # Set image in predictor
#     predictor.set_image(image)
    
#     # Process mask array
#     if mask_array.ndim == 3:
#         mask_array = mask_array[:, :, 0]  # Take first channel if RGB mask
    
#     # Convert to binary mask (0 or 1)
#     mask_array = (mask_array > 128).astype(np.float32)
    
#     # Get image embedding and feature dimensions
#     with torch.no_grad():
#         image_embedding = predictor.get_image_embedding()
#         _, _, h_feat, w_feat = image_embedding.shape
        
#         # Resize mask to match feature dimensions
#         mask_resized = cv2.resize(
#             mask_array,
#             (w_feat, h_feat),
#             interpolation=cv2.INTER_LINEAR
#         )
        
#         # Convert to tensor and reshape
#         mask_tensor = torch.as_tensor(mask_resized, device=device)
#         mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, h_feat, w_feat]
        
#         # Prepare prompt encoder inputs
#         dense_pe = predictor.model.prompt_encoder.get_dense_pe()
        
#         # Create proper sparse embeddings
#         sparse_embeddings = torch.zeros(
#             (1, 0, predictor.model.prompt_encoder.embed_dim),
#             device=device
#         )
        
#         # Universal mask decoder call that works with all SAM versions
#         decoder_output = predictor.model.mask_decoder(
#             image_embeddings=image_embedding,
#             image_pe=dense_pe,
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=mask_tensor,
#             multimask_output=False,
#         )
        
#         # Handle different return signatures
#         if len(decoder_output) == 2:  # Newer SAM versions
#             masks, _ = decoder_output
#         else:  # Older SAM versions (3 returns)
#             masks, _, _ = decoder_output
    
#     # Post-process output mask
#     output_mask = masks[0, 0].cpu().numpy()
#     output_mask = (output_mask > 0).astype(np.uint8)
    
#     # Resize mask back to original image dimensions
#     output_mask = cv2.resize(
#         output_mask,
#         (original_size[1], original_size[0]),
#         interpolation=cv2.INTER_NEAREST
#     )
    
#     return output_mask
