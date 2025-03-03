from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import shutil
import numpy as np
from ultralytics import YOLO
from PIL import Image
import base64
import sys
from PIL import Image
app = Flask(__name__)


# Configure CORS properly with specific origins and methods
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True
    }
})

# Configure upload folder
UPLOAD_FOLDER = '/Users/nehal/KarishmaJewellerayWebsite/model/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_inference(model_path, image_path, output_folder=None):
    """
    Run YOLOv8 segmentation inference on an image and save results
    """
    try:
        # Load the trained YOLOv8 model
        model = YOLO(model_path)
        
        # Set default output folder if none provided
        if output_folder is None:
            output_folder = os.path.join(os.path.dirname(image_path), "runs/segment/predict")
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Run inference on the image
        results = model(image_path, save=True, project=os.path.dirname(output_folder))
        
        # Find the output image in the latest run folder
        latest_run = sorted([f for f in os.listdir(os.path.dirname(output_folder)) 
                           if f.startswith('predict')])[-1]
        output_dir = os.path.join(os.path.dirname(output_folder), latest_run)
        
        if os.path.exists(output_dir):
            output_files = [f for f in os.listdir(output_dir) 
                          if f.endswith(('.jpg', '.png'))]
            
            if output_files:
                output_image = os.path.join(output_dir, output_files[0])
                print(f"✅ Inference completed successfully! Output saved to: {output_image}")
                return results, output_image
            
        print("❌ No output image found in the results folder!")
        return None, None
    
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        return None, None

def extract_components(results, image_path, save_dir="components"):
    try:
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Load image using Pillow
        image = Image.open(image_path).convert('RGB')  # Convert to RGB for consistency
        width, height = image.size  # Get dimensions
        
        # Dictionary to store component counts and images
        component_count = {}
        component_images = {}
        
        # Extract bounding boxes and masks
        for i, result in enumerate(results):
            if not hasattr(result, 'masks') or result.masks is None:
                print("⚠️ No masks found in results")
                continue
                
            for j, mask in enumerate(result.masks.xy):
                # Convert mask to contour
                mask = np.array(mask, dtype=np.int32)
                x_min, y_min = np.min(mask, axis=0)
                x_max, y_max = np.max(mask, axis=0)
                
                # Validate coordinates to ensure they're within image bounds
                x_min = max(0, int(x_min))
                y_min = max(0, int(y_min))
                x_max = min(width, int(x_max))
                y_max = min(height, int(y_max))
                
                # Skip if invalid dimensions
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                # Extract component name (if classes available)
                label = "Component"  # Default label
                if hasattr(result, 'names') and result.names and hasattr(result, 'boxes'):
                    label = result.names[int(result.boxes.cls[j].item())]
                
                # Update count
                if label in component_count:
                    component_count[label] += 1
                else:
                    component_count[label] = 1
                
                # Crop detected component using Pillow
                cropped_component = image.crop((x_min, y_min, x_max, y_max))
                
                # Skip if cropped component is empty
                if cropped_component.size == (0, 0):
                    print(f"⚠️ Skipping empty component: {label}_{component_count[label]}")
                    continue
                
                # Save the cropped component
                component_filename = os.path.join(save_dir, f"{label}_{component_count[label]}.png")
                cropped_component.save(component_filename, 'PNG')
                
                # Store the first detected image of each component type
                if label not in component_images:
                    # Convert to NumPy array for consistency with original code
                    component_images[label] = np.array(cropped_component)
        
        return component_count, component_images
        
    except Exception as e:
        print(f"❌ Error extracting components: {str(e)}")
        return {}, {}
@app.route("/test", methods=["GET"])
def test():
    return jsonify({'message': 'Hello, World!'})

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            # Define all directory paths first
            output_folder = os.path.join(app.config['UPLOAD_FOLDER'], "output")
            components_dir = os.path.join(app.config['UPLOAD_FOLDER'], "components")
            temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], "temp_for_zip")

            # Clean up any existing directories first
            for directory in [UPLOAD_FOLDER, output_folder, components_dir, temp_dir]:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)

            # Save uploaded file
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Model path
            model_path = "/Users/nehal/KarishmaJewellerayWebsite/model/yolov8-seg.torchscript"
            
            # Run inference
            output_folder = os.path.join(app.config['UPLOAD_FOLDER'], "output")
            results, output_path = run_inference(model_path, image_path, output_folder)

            # Modify the response to include both file and analysis data
            if results is not None and output_path:
                # Extract components
                components_dir = os.path.join(app.config['UPLOAD_FOLDER'], "components")
                component_count, _ = extract_components(results, image_path, components_dir)

                # Create ZIP file
                zip_path = os.path.join(app.config['UPLOAD_FOLDER'], "results.zip")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                shutil.make_archive(zip_path[:-4], 'zip', components_dir)

                # Read the segmented image and encode it to base64
                with open(output_path, 'rb') as img_file:
                    segmented_image = base64.b64encode(img_file.read()).decode('utf-8')

                # Read the ZIP file and encode it to base64
                with open(zip_path, 'rb') as zip_file:
                    zip_data = base64.b64encode(zip_file.read()).decode('utf-8')

                # Create response with all data
                response = {
                    'analysis': {
                        'message': f"✅ Found {sum(component_count.values())} components across {len(component_count)} categories",
                        'components': component_count
                    },
                    'segmentedImage': f"data:image/png;base64,{segmented_image}",
                    'zipFile': {
                        'data': f"data:application/zip;base64,{zip_data}",
                        'filename': 'results.zip'
                    }
                }

                # Clean up
                if os.path.exists(image_path):
                    os.remove(image_path)
                for directory in [components_dir, output_folder, temp_dir]:
                    if os.path.exists(directory):
                        shutil.rmtree(directory)
                if os.path.exists(zip_path):
                    os.remove(zip_path)

                return jsonify(response)

            return jsonify({'error': 'Processing failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)