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
            NORMALIZED_MASK_THRESHOLD = 0.3 # Iteration 4: Changed from 0.7 to 0.3 based on user feedback / logs

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
            # --- JULES: Implement Idea A - Select SAM mask based on overlap with user's drawing ---
            if not processed_masks_info:
                print(f"[SAM Debug {image_id_str}] No masks processed from SAM output. Critical fallback.")
                best_mask_np = None # Will trigger fallback later
                selected_mask_info_log = "No masks from SAM to evaluate."
            else:
                # --- JULES: Modify SAM Mask Selection Logic for Object Expansion ---
                # Select SAM mask with the highest internal SAM IoU score.
                best_sam_internal_iou = -1
                selected_mask_index_by_sam_iou = -1

                for idx, p_mask_info in enumerate(processed_masks_info):
                    print(f"[SAM Debug {image_id_str}] Evaluating SAM Mask {p_mask_info['id']} - Internal SAM IoU: {p_mask_info['iou']:.4f}")
                    if p_mask_info['iou'] > best_sam_internal_iou:
                        best_sam_internal_iou = p_mask_info['iou']
                        selected_mask_index_by_sam_iou = idx

                if selected_mask_index_by_sam_iou != -1:
                    best_mask_np = processed_masks_info[selected_mask_index_by_sam_iou]["mask"]
                    selected_mask_original_id = processed_masks_info[selected_mask_index_by_sam_iou]['id']
                    selected_mask_info_log = (f"Selected Mask {selected_mask_original_id} by best internal SAM IoU: {best_sam_internal_iou:.4f}")
                else:
                    # This case should be rare if processed_masks_info is not empty.
                    # If it happens, create a blank mask to prevent downstream errors.
                    sam_output_h, sam_output_w = processed_masks_info[0]["mask"].shape[:2] # Get shape from a sample
                    best_mask_np = np.zeros((sam_output_h, sam_output_w), dtype=np.uint8)
                    selected_mask_info_log = "No SAM mask selected (e.g., processed_masks_info empty or all IoUs were invalid). Using blank mask."
                    print(f"[SAM Debug {image_id_str}] Critical: No SAM mask could be selected based on internal IoU.")
                # --- JULES: Restore SAM Mask Selection Logic based on User Drawing IoU ---
                # Resize user's original drawing (mask_array) to SAM's output mask resolution for comparison
                sam_output_h, sam_output_w = processed_masks_info[0]["mask"].shape[:2]
                user_drawing_resized_for_eval = cv2.resize(mask_array, (sam_output_w, sam_output_h), interpolation=cv2.INTER_NEAREST)
                user_drawing_resized_for_eval = (user_drawing_resized_for_eval > 0.5).astype(np.uint8) # Ensure binary
                # cv2.imwrite(os.path.join(debug_mask_dir, f"user_drawing_resized_for_eval_{image_id_str}.png"), user_drawing_resized_for_eval * 255) # Optional debug

                best_overlap_score = -1
                selected_mask_index_by_overlap = -1

                for idx, p_mask_info in enumerate(processed_masks_info):
                    sam_candidate_mask = p_mask_info["mask"]
                    intersection = np.logical_and(sam_candidate_mask, user_drawing_resized_for_eval).sum()
                    union = np.logical_or(sam_candidate_mask, user_drawing_resized_for_eval).sum()
                    iou_with_user_drawing = intersection / (union + 1e-6)
                    p_mask_info["iou_with_user"] = iou_with_user_drawing
                    print(f"[SAM Debug {image_id_str}] Mask {p_mask_info['id']} (SAM IOU: {p_mask_info['iou']:.4f}) - IoU with User Drawing: {iou_with_user_drawing:.4f}")
                    if iou_with_user_drawing > best_overlap_score:
                        best_overlap_score = iou_with_user_drawing
                        selected_mask_index_by_overlap = idx

                # This part will be handled by the restored fallback logic in the next step.
                # For now, just get best_mask_np based on this user overlap.
                if selected_mask_index_by_overlap != -1:
                    best_mask_np = processed_masks_info[selected_mask_index_by_overlap]["mask"]
                    selected_mask_original_id = processed_masks_info[selected_mask_index_by_overlap]['id']
                    sam_internal_iou = processed_masks_info[selected_mask_index_by_overlap]['iou']
                    selected_mask_info_log = (f"Selected Mask {selected_mask_original_id} by User Drawing Overlap "
                                              f"(User IoU: {best_overlap_score:.4f}, SAM IoU: {sam_internal_iou:.4f})")
                else:
                    # Fallback if no positive overlap, pick by SAM's best internal IoU as a last resort before outer fallback.
                    processed_masks_info.sort(key=lambda x: x["iou"], reverse=True)
                    if processed_masks_info: # Check if list is not empty
                        best_mask_np = processed_masks_info[0]["mask"]
                        selected_mask_info_log = (f"No positive user overlap. Tentatively selected Mask {processed_masks_info[0]['id']} "
                                                  f"by SAM IOU: {processed_masks_info[0]['iou']:.4f}")
                    else: # Should not happen if initial check for processed_masks_info passed
                        best_mask_np = np.zeros((sam_output_h, sam_output_w), dtype=np.uint8) # Default blank
                        selected_mask_info_log = "Critical: No masks in processed_masks_info for selection."

            # --- END JULES Restored User Drawing IoU Selection ---

            # --- JULES: Restore Fallback Logic ---
            MIN_USER_IOU_FALLBACK_THRESHOLD = 0.10 # Lowered from 0.25
            did_fallback_to_user_drawing = False

            # Check if a mask was selected by user overlap and if its score is sufficient
            if selected_mask_index_by_overlap != -1 and best_overlap_score >= MIN_USER_IOU_FALLBACK_THRESHOLD:
                # SAM's mask is good enough (based on user drawing overlap)
                # best_mask_np is already set from the selection logic above.
                # selected_mask_info_log is also already set.
                did_fallback_to_user_drawing = False
                print(f"[SAM Debug {image_id_str}] SAM Mask chosen (User IoU {best_overlap_score:.4f} >= {MIN_USER_IOU_FALLBACK_THRESHOLD}).")
            else:
                # Fallback to user's original drawing
                best_mask_np = user_drawing_resized_for_eval # Use user's drawing (resized to SAM output resolution)
                did_fallback_to_user_drawing = True
                if selected_mask_index_by_overlap != -1: # A SAM mask was technically best by overlap, but score was too low
                     selected_mask_info_log = (f"FALLBACK to user's drawing. Best SAM mask had User IoU {best_overlap_score:.4f} "
                                               f"(< {MIN_USER_IOU_FALLBACK_THRESHOLD}). Using user's drawn mask directly.")
                elif best_sam_internal_iou > 0 : # No positive overlap with user drawing, but SAM had a best guess
                    selected_mask_info_log = (f"FALLBACK to user's drawing. No positive overlap with user input. SAM's best internal IoU was {best_sam_internal_iou:.4f}. Using user's drawn mask directly.")
                else: # No positive overlap AND SAM had no best guess (should be rare)
                     selected_mask_info_log = (f"FALLBACK to user's drawing. No SAM mask suitable. Using user's drawn mask directly.")
                best_mask_np = best_mask_np.astype(np.uint8)
            # --- END JULES Restore Fallback Logic ---

        print(f"[SAM Debug {image_id_str}] {selected_mask_info_log} to pass to LaMa.")

        if best_mask_np is None:
            print(f"[SAM Debug {image_id_str}] Critical: best_mask_np is None after selection/fallback. Using blank mask.")
            # Determine fallback shape more robustly
            if 'sam_output_h' in locals() and 'sam_output_w' in locals() and sam_output_h > 0 and sam_output_w > 0:
                fallback_h, fallback_w = sam_output_h, sam_output_w
            elif masks_from_sam is not None and masks_from_sam.shape[2] > 0 and masks_from_sam.shape[3] > 0:
                 fallback_h, fallback_w = masks_from_sam.shape[2], masks_from_sam.shape[3]
            else:
                 fallback_h, fallback_w = 256, 256 # Absolute last resort
            best_mask_np = np.zeros((fallback_h, fallback_w), dtype=np.uint8)

        # --- JULES: Restore Conditional Inversion and Constraining Logic ---
        if not did_fallback_to_user_drawing:
            # SAM's mask was chosen and is good enough.
            # DO NOT INVERT best_mask_np. It already represents the object to inpaint.
            # inverted_best_mask_np = 1 - best_mask_np # REMOVE THIS LINE
            # print(f"[SAM Debug {image_id_str}] Inverted the selected SAM mask. Unique values: {np.unique(inverted_best_mask_np)}") # REMOVE/COMMENT

            # Constrain SAM's output with the user's drawing
            # Ensure sam_output_h, sam_output_w are from best_mask_np for safety
            sam_output_h, sam_output_w = best_mask_np.shape[:2] # Use best_mask_np directly
            user_drawing_resized_for_constrain = cv2.resize(mask_array, (sam_output_w, sam_output_h), interpolation=cv2.INTER_NEAREST)
            user_drawing_resized_for_constrain = (user_drawing_resized_for_constrain > 0.5).astype(np.uint8)
            print(f"[SAM Debug {image_id_str}] Resized user drawing for constraining. Shape: {user_drawing_resized_for_constrain.shape}, Unique: {np.unique(user_drawing_resized_for_constrain)}")

            # Use best_mask_np directly for constraining
            constrained_mask_at_sam_res = best_mask_np * user_drawing_resized_for_constrain 
            print(f"[SAM Debug {image_id_str}] Constrained SAM mask (best_mask_np * user_drawing) with user drawing. Shape: {constrained_mask_at_sam_res.shape}, Unique: {np.unique(constrained_mask_at_sam_res)}")
        else:
            # Fallback occurred: use the user's drawing directly.
            # best_mask_np is already user_drawing_resized_for_eval.
            # LaMa expects 1 where inpainting should occur.
            constrained_mask_at_sam_res = best_mask_np 
            print(f"[SAM Debug {image_id_str}] Fallback: Using user's drawing directly as mask for LaMa. Shape: {constrained_mask_at_sam_res.shape}, Unique: {np.unique(constrained_mask_at_sam_res)}")
        # --- END JULES Restore Conditional Inversion and Constraining Logic ---

        # Resize the final mask to original image size
        final_mask_output = cv2.resize(constrained_mask_at_sam_res, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        print(f"[SAM Debug {image_id_str}] Final resized (constrained and inverted) mask output - Shape: {final_mask_output.shape}, Unique values: {np.unique(final_mask_output)}")
        # Save the final chosen mask that will be used by LaMa for clarity
        cv2.imwrite(os.path.join(debug_mask_dir, f"sam_final_chosen_multimask_{image_id_str}.png"), final_mask_output * 255)

    return final_mask_output
