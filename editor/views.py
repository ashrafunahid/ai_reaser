import os
import base64
import numpy as np
import cv2
from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
from .sam_segment import generate_mask_with_point, generate_mask_with_mask
from .yolo_segment import generate_mask_with_yolo
from .lama_infer import load_lama_model, run_lama_inpainting
# from .zits_infer import run_zits_inpainting
from .zitspp_infer import run_zitspp

# Declaring Model Globally
try:
    # lama_model = load_lama_model() # Old
    # lama_model, lama_refinement_kwargs = load_lama_model() # Previous version
    lama_model, lama_refiner_config = load_lama_model() # New variable name
except Exception as e:
    print("Failed to load LaMa Model", e)
    lama_model, lama_refiner_config = None, None
    
# cfg_path =  'ZITS_PlusPlus/configs/config_zitspp_finetune.yml',
# ckpt_path = 'ZITS_PlusPlus/ckpts/model_512/models/last.ckpt'

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            
            # If LaMa model not loaded properly then show the error
            if not lama_model:
                return render(request, 'editor/upload.html', {
                    'form': form, 
                    'error': 'Model Failed to load, Please try again'
                    })
                
            uploaded = form.save()
            image_path = uploaded.image.path

            action = request.POST.get('action')  # Check which button was pressed
            x = request.POST.get('x_point')
            y = request.POST.get('y_point')
            mask_data = request.POST.get('mask_data')

            if action == 'mask' and mask_data:
                # Decode the base64 PNG mask into NumPy array
                mask_data = mask_data.split(',')[1]
                mask_bytes = base64.b64decode(mask_data)
                mask_array = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE) 
                mask_array = (mask_array > 128).astype(np.uint8)  # Binary mask

                # --- Debug: Save the mask as processed in views.py ---
                debug_mask_dir = os.path.join('media', 'debug')
                os.makedirs(debug_mask_dir, exist_ok=True)
                # cv2.imwrite(os.path.join(debug_mask_dir, f'view_mask_{uploaded.id}.png'), mask_array * 255)

                segmented_mask = generate_mask_with_mask(image_path, mask_array)

            elif action == 'point' and x and y:
                input_point = [int(x), int(y)]
                segmented_mask = generate_mask_with_point(image_path, input_point)

            else:
                return render(request, 'editor/upload.html', {
                    'form': form,
                    'error': 'Please click (for point) or draw (for mask) on the image!'
                })
                
            yolo_segmented_mask = generate_mask_with_yolo(image_path, mask_array)

            # Save or pass segmented mask to template
            sam_mask_save_path = os.path.join('media', 'outputs', f'sam_mask_{uploaded.id}.png')
            os.makedirs(os.path.dirname(sam_mask_save_path), exist_ok=True)
            cv2.imwrite(sam_mask_save_path, segmented_mask * 255)
            
            yolo_mask_save_path = os.path.join('media', 'outputs', f'yolo_mask_{uploaded.id}.png')
            os.makedirs(os.path.dirname(yolo_mask_save_path), exist_ok=True)
            cv2.imwrite(yolo_mask_save_path, yolo_segmented_mask * 255)
            
            # Inpaint with LaMa
            # inpainted = run_lama_inpainting(image_path, mask_save_path, lama_model) # Old
            # inpainted = run_lama_inpainting(image_path, mask_save_path, lama_model, lama_refinement_kwargs) # Previous
            inpainted = run_lama_inpainting(image_path, sam_mask_save_path, lama_model, lama_refiner_config) # New
            # inpainted_zits = run_zits_inpainting(image_path, sam_mask_save_path)
            inpainted_zitspp = run_zitspp(image_path, sam_mask_save_path, 'ZITS_PlusPlus/ckpts/model_512/models/last.ckpt')
            inpainted_yolo = run_lama_inpainting(image_path, yolo_mask_save_path, lama_model, lama_refiner_config) 

            # Save inpainted result
            inpaint_path_lama = os.path.join('media', 'outputs', f'inpaint_lama_{uploaded.id}.png')
            inpaint_path_zits = os.path.join('media', 'outputs', f'inpaint_zits_{uploaded.id}.png')
            inpaint_path_zitspp = os.path.join('media', 'outputs', f'inpaint_zitspp_{uploaded.id}.png')
            inpaint_path_yolo = os.path.join('media', 'outputs', f'yolo_inpaint_{uploaded.id}.png')
            # --- Convert RGB to BGR for cv2.imwrite ---
            inpainted_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)
            # inpainted_zits_bgr = cv2.cvtColor(inpainted_zits, cv2.COLOR_RGB2BGR)
            inpainted_zitspp_bgr = cv2.cvtColor(inpainted_zitspp, cv2.COLOR_RGB2BGR)
            inpainted_bgr_yolo = cv2.cvtColor(inpainted_yolo, cv2.COLOR_RGB2BGR)
            cv2.imwrite(inpaint_path_lama, inpainted_bgr)
            # cv2.imwrite(inpaint_path_zits, inpainted_zits_bgr)
            cv2.imwrite(inpaint_path_zitspp, inpainted_zitspp_bgr)
            cv2.imwrite(inpaint_path_yolo, inpainted_bgr_yolo)

            return render(request, 'editor/result.html', {
                'image': uploaded,
                'mask_path': '/' + sam_mask_save_path,
                'inpainted_path_lama': '/' + inpaint_path_lama,
                'inpainted_path_yolo': '/' + inpaint_path_yolo,
                'inpainted_path_zits': '/' + inpaint_path_zits,
                'inpainted_path_zitspp': '/' + inpaint_path_zitspp,
            })

    else:
        form = ImageUploadForm()

    return render(request, 'editor/upload.html', {'form': form})

