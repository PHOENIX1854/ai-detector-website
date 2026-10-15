import os
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import detector_logic # This is our refactored Python logic

# --- 1. Configuration ---
app = Flask(__name__)
# Allow our website (running on a different "origin") to talk to this API
CORS(app) 

# Create an 'uploads' folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- 2. Load Models ONCE ---
# We load the models into memory when the server starts.
# This is *much* faster than loading them for every request.
print("--- Server is starting ---")
print("Loading AI models... (This may take a moment)")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
try:
    VLM_MODEL, PROCESSOR, CLASSIFIER_MODEL = detector_logic.load_models(device)
    print("--- Models loaded successfully. Server is ready. ---")
except Exception as e:
    print(f"FATAL ERROR: Could not load models: {e}")
    print("Please make sure 'best_nn_classifier_FINAL.pth' is in the same directory.")
    # In a real app, you'd exit, but we'll let it run so the user sees the error
    VLM_MODEL, PROCESSOR, CLASSIFIER_MODEL = None, None, None

# --- 3. Define API Endpoint ---
@app.route('/api/detect', methods=['POST'])
def detect_file():
    if VLM_MODEL is None:
        # Return a 500 Internal Server Error
        return jsonify({"error": "Models are not loaded. Check server logs."}), 500
        
    # Check if a file was sent
    if 'media_file' not in request.files:
        return jsonify({"error": "No 'media_file' part in request"}), 400
    
    file = request.files['media_file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)
        
        result = {}
        try:
            # Check file extension to decide which function to use
            ext = os.path.splitext(file.filename)[1].lower()
            
            if ext in detector_logic.IMAGE_EXTENSIONS_SET:
                result = detector_logic.get_prediction_for_image_file(
                    temp_path, VLM_MODEL, PROCESSOR, CLASSIFIER_MODEL, device
                )
            elif ext in detector_logic.VIDEO_EXTENSIONS_SET:
                result = detector_logic.process_video_file(
                    temp_path, VLM_MODEL, PROCESSOR, CLASSIFIER_MODEL, device
                )
            else:
                result = {"error": f"Unsupported file type: {ext}"}
                
        except Exception as e:
            print(f"Error during processing: {e}")
            result = {"error": f"An error occurred during processing: {e}"}
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Send the result back to the JavaScript
        return jsonify(result)

# --- 4. Serve the HTML & JS Files ---
# This makes it easy to run. Just go to http://127.0.0.1:5000

@app.route('/')
def serve_index():
    # Serves the ai_detector.html file
    return send_from_directory('.', 'ai_detector.html')

@app.route('/ai_detector.js')
def serve_js():
    # Serves the ai_detector.js file
    return send_from_directory('.', 'ai_detector.js')

# --- 5. Run the Server ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
