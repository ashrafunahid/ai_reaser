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
        mask_resized = (mask_resized > 0.5).astype(np.float32)
        print(f"[SAM Debug {image_id_str}] Binarized mask_resized. Shape: {mask_resized.shape}, Dtype: {mask_resized.dtype}, Unique values: {np.unique(mask_resized)}")
        # --- END JULES ---
        cv2.imwrite(os.path.join(debug_mask_dir, f"sam_mask_resized_{image_id_str}.png"), (mask_resized * 255).astype(np.uint8)) # Save the binarized version for inspection
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
            multimask_output=True # Changed to True
        )

        # outputs[0] are the masks, outputs[1] are the IoU predictions
        # masks_from_sam shape: (batch_size, num_masks, H, W), e.g., (1, 3, 256, 256)
        # iou_predictions shape: (batch_size, num_masks), e.g., (1, 3)
        masks_from_sam, iou_predictions = outputs[0], outputs[1]

        print(f"[SAM Debug {image_id_str}] multimask_output=True. Received {masks_from_sam.shape[1]} masks.")

        processed_masks_info = [] # To store (binary_mask, iou_score, max_logit, original_index)

        for i in range(masks_from_sam.shape[1]):
            single_mask_logits = masks_from_sam[0, i].cpu().numpy()
            iou_score = iou_predictions[0, i].cpu().numpy()
            current_max_logit = single_mask_logits.max() # Keep for logging
            print(f"[SAM Debug {image_id_str}] Mask {i} - IOU: {iou_score:.4f}, Logits Min: {single_mask_logits.min():.4f}, Max: {current_max_logit:.4f}")

            # --- JULES: Universal Normalized Thresholding ---
            NORMALIZED_MASK_THRESHOLD = 0.7 # Iteration 3: Changed from 0.6 to 0.7

            min_logit_norm = single_mask_logits.min()
            max_logit_norm = single_mask_logits.max()

            if max_logit_norm == min_logit_norm:
                normalized_single_mask = np.zeros_like(single_mask_logits)
            else:
                normalized_single_mask = (single_mask_logits - min_logit_norm) / (max_logit_norm - min_logit_norm + 1e-6)

            # Save the normalized view before thresholding
            cv2.imwrite(os.path.join(debug_mask_dir, f"sam_multi_normalized_{image_id_str}_mask{i}.png"), (normalized_single_mask * 255).astype(np.uint8))

            binary_single_mask = (normalized_single_mask > NORMALIZED_MASK_THRESHOLD).astype(np.uint8)
            print(f"[SAM Debug {image_id_str}] Mask {i} - Universal normalized thresholding applied (>{NORMALIZED_MASK_THRESHOLD})")
            # --- END JULES ---

            cv2.imwrite(os.path.join(debug_mask_dir, f"sam_multi_thresholded_{image_id_str}_mask{i}.png"), binary_single_mask * 255)
            processed_masks_info.append({
                "mask": binary_single_mask,
                "iou": iou_score,
                # "max_logit": current_max_logit, # Not directly used for selection anymore but good for info
                "id": i
            })

        # Mask Selection Logic: Select by best IOU from universally processed masks
        best_mask_np = None
        selected_mask_info_log = "No suitable mask found initially."

        if processed_masks_info:
            processed_masks_info.sort(key=lambda x: x["iou"], reverse=True) # Sort by IOU descending
            best_mask_np = processed_masks_info[0]["mask"]
            selected_mask_info_log = f"Selected Mask {processed_masks_info[0]['id']} (Universal Norm, best IOU: {processed_masks_info[0]['iou']:.4f})"

        print(f"[SAM Debug {image_id_str}] {selected_mask_info_log} to pass to LaMa.")

        if best_mask_np is None: # Should ideally not happen if processed_masks_info is populated
            fallback_h, fallback_w = masks_from_sam.shape[2], masks_from_sam.shape[3]
            best_mask_np = np.zeros((fallback_h, fallback_w), dtype=np.uint8)
            print(f"[SAM Debug {image_id_str}] Critical fallback: No mask selected, using blank mask.")

        # Invert the selected mask
        inverted_best_mask_np = 1 - best_mask_np
        print(f"[SAM Debug {image_id_str}] Inverted the selected mask. Unique values after inversion: {np.unique(inverted_best_mask_np)}")

        # --- JULES: Implement Mask Constraining ---
        # inverted_best_mask_np is at SAM's output resolution (e.g., 256x256)
        # mask_array is the original user drawing (from canvas resolution, already binarized 0 or 1)

        sam_output_h, sam_output_w = inverted_best_mask_np.shape[:2]

        # Resize user's original drawing to SAM's output resolution
        # input `mask_array` to this function is already float32 with 0s and 1s.
        user_drawing_resized_for_constrain = cv2.resize(mask_array, (sam_output_w, sam_output_h), interpolation=cv2.INTER_NEAREST)
        # Ensure it's strictly binary after resize, as INTER_NEAREST should keep it but good to be sure.
        user_drawing_resized_for_constrain = (user_drawing_resized_for_constrain > 0.5).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_mask_dir, f"user_drawing_resized_for_constrain_{image_id_str}.png"), user_drawing_resized_for_constrain * 255)
        print(f"[SAM Debug {image_id_str}] Resized user drawing for constraining. Shape: {user_drawing_resized_for_constrain.shape}, Unique: {np.unique(user_drawing_resized_for_constrain)}")

        # Constrain SAM's output with the user's drawing
        # Both are binary (0 or 1), so simple multiplication works as AND
        constrained_mask_at_sam_res = inverted_best_mask_np * user_drawing_resized_for_constrain
        cv2.imwrite(os.path.join(debug_mask_dir, f"sam_constrained_unresized_{image_id_str}.png"), constrained_mask_at_sam_res * 255)
        print(f"[SAM Debug {image_id_str}] Constrained SAM mask with user drawing. Shape: {constrained_mask_at_sam_res.shape}, Unique: {np.unique(constrained_mask_at_sam_res)}")
        # --- END JULES ---

        # Resize the constrained mask to original image size
        final_mask_output = cv2.resize(constrained_mask_at_sam_res, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        print(f"[SAM Debug {image_id_str}] Final resized (constrained and inverted) mask output - Shape: {final_mask_output.shape}, Unique values: {np.unique(final_mask_output)}")
        # Save the final chosen mask that will be used by LaMa for clarity
        cv2.imwrite(os.path.join(debug_mask_dir, f"sam_final_chosen_multimask_{image_id_str}.png"), final_mask_output * 255)

    return final_mask_output


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
