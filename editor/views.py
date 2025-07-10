import os
import base64
import numpy as np
import cv2
from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
from .sam_segment import generate_mask_with_point, generate_mask_with_mask
from .lama_infer import load_lama_model, run_lama_inpainting

# Declaring Model Globally
# lama_model = load_lama_model() # Old
# lama_model, lama_refinement_kwargs = load_lama_model() # Previous version
lama_model, lama_refiner_config = load_lama_model() # New variable name

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
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
                cv2.imwrite(os.path.join(debug_mask_dir, f'view_mask_{uploaded.id}.png'), mask_array * 255)

                segmented_mask = generate_mask_with_mask(image_path, mask_array)

            elif action == 'point' and x and y:
                input_point = [int(x), int(y)]
                segmented_mask = generate_mask_with_point(image_path, input_point)

            else:
                return render(request, 'editor/upload.html', {
                    'form': form,
                    'error': 'Please click (for point) or draw (for mask) on the image!'
                })

            # Save or pass segmented mask to template
            mask_save_path = os.path.join('media', 'outputs', f'mask_{uploaded.id}.png')
            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
            cv2.imwrite(mask_save_path, segmented_mask * 255)
            
            # Inpaint with LaMa
            # inpainted = run_lama_inpainting(image_path, mask_save_path, lama_model) # Old
            # inpainted = run_lama_inpainting(image_path, mask_save_path, lama_model, lama_refinement_kwargs) # Previous
            inpainted = run_lama_inpainting(image_path, mask_save_path, lama_model, lama_refiner_config) # New

            # Save inpainted result
            inpaint_path = os.path.join('media', 'outputs', f'inpaint_{uploaded.id}.png')
            # --- Convert RGB to BGR for cv2.imwrite ---
            inpainted_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)
            cv2.imwrite(inpaint_path, inpainted_bgr)

            return render(request, 'editor/result.html', {
                'image': uploaded,
                'mask_path': '/' + mask_save_path,
                'inpainted_path': '/' + inpaint_path
            })

    else:
        form = ImageUploadForm()

    return render(request, 'editor/upload.html', {'form': form})

