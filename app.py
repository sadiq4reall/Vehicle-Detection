from flask import Flask, request, jsonify, send_file, send_from_directory
import os
import cv2
import uuid
import torch
import torchvision.transforms as T
from PIL import Image

# Import from existing project structure
try:
    from config import DEVICE, NUM_CLASSES, CLASSES, YOLO_MODEL, YOLO_DEVICE
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import torchvision
except ImportError:
    print("Warning: Could not import configuration. Ensure config.py is in the root directory.")

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# -----------------
# Helper Logic
# -----------------
def load_rcnn_model():
    """Dynamically loads the Faster R-CNN model using PyTorch architecture."""
    weights_path = os.path.join(os.getcwd(), 'fasterrcnn_car_detector.pth')
    if not os.path.exists(weights_path):
        return None
        
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    
    device = torch.device(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_yolo_model():
    """Loads the YOLOv11 model cleanly."""
    root_weights = os.path.join(os.getcwd(), 'best.pt')
    yolo_weights = os.path.join(os.getcwd(), 'runs', 'detect', 'train', 'weights', 'best.pt')
    
    if os.path.exists(root_weights):
        yolo_weights = root_weights
    elif not os.path.exists(yolo_weights):
        import urllib.request
        print(f"Downloading model from GitHub to {yolo_weights}...")
        os.makedirs(os.path.dirname(yolo_weights), exist_ok=True)
        url = "https://github.com/alawalmuazu/Vehicle-Detection/raw/main/runs/detect/train/weights/best.pt"
        urllib.request.urlretrieve(url, yolo_weights)
    from ultralytics import YOLO
    return YOLO(yolo_weights)


# -----------------
# Routes
# -----------------
@app.route('/')
def index():
    return send_file('templates/index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400

    architecture = request.form.get('architecture', 'yolo')
    confidence = float(request.form.get('confidence', 0.25))

    # Save incoming image safely
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    output_filename = f"detected_{filename}"
    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    predictions = []

    try:
        if architecture == 'yolo':
            model = get_yolo_model()
            results = model(filepath, conf=confidence)
            
            res = results[0]
            im_bgr = res.plot()
            cv2.imwrite(output_filepath, im_bgr)
            
            boxes = res.boxes.xyxy.cpu().numpy()
            labels = res.boxes.cls.cpu().numpy().astype(int)
            scores = res.boxes.conf.cpu().numpy()
            names = model.names
            
            for (x1, y1, x2, y2), c, s in zip(boxes, labels, scores):
                predictions.append({
                    "class": names[c],
                    "confidence": float(s),
                    "box": [float(x1), float(y1), float(x2), float(y2)]
                })

        elif architecture == 'rcnn':
            model = load_rcnn_model()
            if not model:
                return jsonify({"error": "Faster R-CNN weights not found. Please train first."}), 500
                
            image = Image.open(filepath).convert('RGB')
            transform = T.ToTensor()
            device = torch.device(DEVICE)
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)

            # Extract results natively
            pred_boxes = outputs[0]['boxes'].cpu().numpy()
            pred_scores = outputs[0]['scores'].cpu().numpy()
            
            # Map over OpenCV
            im_cv = cv2.imread(filepath)
            for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                if score >= confidence:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(im_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Vehicle: {score:.2f}"
                    cv2.putText(im_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    predictions.append({
                        "class": "Vehicle",
                        "confidence": float(score),
                        "box": [float(x1), float(y1), float(x2), float(y2)]
                    })
            
            cv2.imwrite(output_filepath, im_cv)

        return jsonify({
            "success": True,
            "architecture": architecture,
            "count": len(predictions),
            "predictions": predictions,
            "output_image_url": f"/outputs/{output_filename}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/outputs/<filename>')
def get_output_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    print("[OK] Starting Web UI on http://localhost:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
